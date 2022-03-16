import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.network import *
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

seed = int(0)
n_neurons = int(400)
n_epochs = int(1)
n_test = int(10000)
n_train = int(60000)
n_workers = int(-1)
exc = float(22.5)
inh = float(120)
theta_plus = float(0.05)
time = int(250)
dt = int(1.0)
intensity = float(128)
progress_interval = int(10)
update_interval = int(250)
train = False
plot = True
gpu = True

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity



# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

network = load('../BD400/network400.pt', device)
network.reset_state_variables()  # Reset state variables.

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt)
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt)
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = torch.load('../BD400/assignments400.pt',device)
proportions = torch.load('../BD400/proportions400.pt',device)
    
# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

epoch = 1

contSample = int(0)

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    inputNetwork = inputs['X'][:,0,0,:,:].cpu().numpy()
    filename_InSamp = '../inputSamples/{0:05d}_inputSpikesPoisson'.format(step+1)+'.csv'
    inputNetwork.tofile( filename_InSamp , sep = ',' )
    
    '''
    print('{0:=>30}'.format(''))
    print("\nstep:" , step+1, " sample: ", batch['encoded_label'])
    print('{0:=>30}'.format(''))
    '''
    network.run(inputs=inputs, time=time, input_time_dim=1)
    
    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)
    with open( '../classification/labelsBindsnetIn.csv', 'a', encoding='utf-8' ) as f:
        f.write( str( int(label_tensor) ) +'\n' )

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    
    with open( '../classification/labelsBindsnetOut.csv', 'a', encoding='utf-8' ) as f:
        f.write( str( int(all_activity_pred) ) +'\n' )
        
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")