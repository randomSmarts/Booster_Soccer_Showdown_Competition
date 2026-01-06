import torch
import torch.nn as nn
import os
from sac import SAC
import numpy as np

# This file extracts the robot movements and saves it into a format called TorchScript (.pth)

# --- CONFIG ---
CHECKPOINT_PATH = "balance_models/sac_balance_checkpoint.pth" # Path to your best model 
# TODO need to change
OUTPUT_PATH = "balance_models/submitA.pth" # What you will actually submit 
# TODO need to change

STATE_DIM = 88  # Must match your Preprocessor output, this is the # of parameters the robot sees
# DIM = dimensions
ACTION_DIM = 12 #The number of parameters the robot controls 

NEURONS = [256, 256] # Must match your SAC config exactly since this is the size of the brain!

class PrimitivePolicy(nn.Module): #this is the clean model/brain using standard module template
    def __init__(self, n_in, n_out, neurons): #n_in is STATE_DIM (inputs), n_out is ACTION_DIM (robot motion), neurons is the size of each layer
        super().__init__()
        layers = [] #stores each layer in the brain
        curr_in = n_in #stores the amount of wires coming from the prev. layer, starting is 88
        for h in neurons: #iterates through the brain
            # TODO: figure out how the next two layers conceptually work
            layers.append(nn.Linear(curr_in, h)) #creates the math layer 
            layers.append(nn.ReLU()) #creates the logic layer
            curr_in = h #updates the amount of wires coming from prev. level
        self.backbone = nn.Sequential(*layers) #creates one backbone from all of the layers
        self.output_layer = nn.Linear(curr_in, n_out) #takes the output of the backbone and tranforms it into the 12 numbesr for the robot joints

    def forward(self, x): #runs when the robot is playing
        # x is what the robot sees
        x = x.float() #force input to float before hitting first linear layer
        x = self.backbone(x)
        x = self.output_layer(x)
        return torch.tanh(x) #after passing in the input it runs it through the layers and returns the needed numbers, albeit bounded so the robot doesn't do anything with 10000% power

def export():
    # 1. Load your actual trained SAC agent
    class DummyActionSpace:
        def __init__(self): self.shape = (ACTION_DIM,)
    
    trained_agent = SAC(n_features=STATE_DIM, action_space=DummyActionSpace(), neurons=NEURONS, device="cpu") #empty, dirty SAC agent
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    trained_agent.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu")) #inputs all of the learned knowledge from trained model
    
    # 2. Create the Primitive "Clone"
    primitive_model = PrimitivePolicy(STATE_DIM, ACTION_DIM, NEURONS) #empty, clean version of brain
    
    # 3. Manually copy the weights layer-by-layer
    # This is the 'clean' way to move the brain without the 'baggage'
    print("Cloning weights to primitive model...")
    
    # Copy Backbone (Sequential linear layers)
    # trained_agent.actor.base_net[0] -> primitive_model.backbone[0]
    # trained_agent.actor.base_net[2] -> primitive_model.backbone[2]
    with torch.no_grad(): #tells torch to only copy the data and not do the math
        # Copy base_net layers (they are at indices 0, 2, 4...)
        for i in range(len(NEURONS)): #iterates through the size of the brain (# of layers) and copies the weights and biases
            idx = i * 2
            # TODO: figure out what the weights and biases actually mean adn why idx = i*2
            primitive_model.backbone[idx].weight.copy_(trained_agent.actor.base_net[idx].weight) # type: ignore
            primitive_model.backbone[idx].bias.copy_(trained_agent.actor.base_net[idx].bias) # type: ignore
            
        # Copy Final Output (Mean)
        primitive_model.output_layer.weight.copy_(trained_agent.actor.mean_linear.weight)
        primitive_model.output_layer.bias.copy_(trained_agent.actor.mean_linear.bias)

    primitive_model.eval() #sets the model to work mode now
    
    # 4. Trace the Primitive Model (seeing if all is good with fake piece of data)
    example_input = torch.randn(1, STATE_DIM).float() * 0.5
    print("Tracing primitive model...")
    try:
        # TODO: understand what this is --> We use a simple lambda to ensure absolute purity
        traced_model = torch.jit.trace(primitive_model, example_input) #trace = watches how data flows through the model and creates a static map
        torch.jit.save(traced_model, OUTPUT_PATH) #exports the static map (which is what runs on SAI servers since it is a fixed set of instructions for all dynamic inputs)
        print(f"✅ CLEAN EXPORT SUCCESSFUL: {OUTPUT_PATH}")
        
        # Verify it works
        reloaded = torch.jit.load(OUTPUT_PATH)
        out = reloaded(example_input)
        print(f"Verification Output: {out[0, :3]}... (Action matches)") #runs fake data through the static map and if it outputs three numbers then its good

    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export()

