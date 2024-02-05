
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from pathlib import Path


class FairTransformerGAN(nn.Module):
    def __init__(self, dataType='binary', inputDim=58, embeddingDim=32, randomDim=32,
                 generatorDims=(32, 32), discriminatorDims=(32, 16, 1), compressDims=(),
                 decompressDims=(), bnDecay=0.99, l2scale=0.001, lambda_fair=1):
        super(FairTransformerGAN, self).__init__()

        # Model configuration
        self.dataType = dataType
        self.inputDim = inputDim
        self.embeddingDim = embeddingDim
        self.randomDim = randomDim
        self.generatorDims = generatorDims
        self.discriminatorDims = discriminatorDims
        self.compressDims = compressDims
        self.decompressDims = decompressDims
        self.bnDecay = bnDecay
        self.l2scale = l2scale
        self.lambda_fair = lambda_fair

        # Model components
        self.generator = self.buildGenerator()
        self.discriminator = self.buildDiscriminator()
        self.autoencoder = self.buildAutoencoder()

    def loadData(self, dataPath='/Users/penghuizhu/Desktop/Workspace/FairGAN/examples/csv/adult.csv'):
        # Load the data from a NumPy file
        data = np.load(dataPath)

        # Assuming data structure: features (X), protected attributes (z), and targets (y)
        X = data['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']   # Feature matrix
        z = data['sex']  # Protected attributes
        y = data['income'] # Targets or labels

        # Split the data into train and validation sets
        trainX, validX, trainy, validy = train_test_split(X, y, test_size=0.2, random_state=42)
        trainz, validz = train_test_split(z, test_size=0.2, random_state=42)

        return trainX, validX, trainz, validz, trainy, validy

# Example usage:
# model = FairTransformerGAN()
# trainX, validX, trainz, validz, trainy, validy = model.loadData(dataPath='path_to_your_data_file.npy')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Assuming the input data is normalized between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def buildAutoencoder(x_input, input_dim):
    model = Autoencoder(input_dim)
    output = model(x_input)

    # Assuming the use of MSE Loss for reconstruction
    criterion = nn.MSELoss()
    loss = criterion(output, x_input)

    # Collecting weights and biases from the decoder
    decodeVariables = {'weights': [], 'biases': []}
    for layer in model.decoder:
        if isinstance(layer, nn.Linear):
            decodeVariables['weights'].append(layer.weight.data.numpy())
            decodeVariables['biases'].append(layer.bias.data.numpy())

    return loss, decodeVariables


# Example usage:
# Assume x_input is your input tensor with shape [batch_size, input_dim]
# For demonstration, let's create a random tensor as input
batch_size, input_dim = 64, 784  # Example input dimensions
x_input = torch.randn(batch_size, input_dim)

# Build the autoencoder and calculate loss
loss, decodeVariables = buildAutoencoder(x_input, input_dim)
print("Reconstruction Loss:", loss.item())
# Printing out the first layer's weights and biases sizes for verification
print("First layer weights size:", decodeVariables['weights'][0].shape)
print("First layer biases size:", decodeVariables['biases'][0].shape)

class MultiHeadSelfAttention(nn.Module):
    # Placeholder for the multi-head self-attention mechanism.
    # Implement this class based on your specific needs or use an existing implementation.
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        # Implementation details here...

    def forward(self, x):
        # Implementation details here...
        return x


class Generator(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, embed_size, heads, output_dim):
        super(Generator, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.output_dim = output_dim

        # Example linear layer to combine inputs
        self.combine_inputs = nn.Linear(x_dim + y_dim + z_dim, embed_size)

        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embed_size, heads)

        # Batch normalization
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Final layer to generate output
        self.output_layer = nn.Linear(embed_size, output_dim)

    def forward(self, x_input, y_input, z_input, bn_train=True):
        # Concatenate input tensors
        combined_input = torch.cat((x_input, y_input, z_input), dim=1)

        # Pass through the initial combination layer
        x = self.combine_inputs(combined_input)

        # Apply multi-head self-attention
        x = self.attention(x)

        # Apply batch normalization
        x = self.bn(x) if bn_train else x

        # Generate output
        output = self.output_layer(x)

        return output

# Example usage
x_dim = 10
y_dim = 1
z_dim = 1
embed_size = 128
heads = 4
output_dim = 10

generator = Generator(x_dim, y_dim, z_dim, embed_size, heads, output_dim)

# Example tensors
x_input = torch.randn(32, x_dim)
y_input = torch.randn(32, y_dim)
z_input = torch.randn(32, z_dim)
bn_train = True

output = generator(x_input, y_input, z_input, bn_train)
print(output.shape)  # Expected shape: [32, output_dim]


class Discriminator(nn.Module):
    def __init__(self, input_dim, protected_dim, output_dim=1, hidden_dims=(32, 16)):
        super(Discriminator, self).__init__()
        # Initial input_dim adjustment to account for concatenated protected attributes
        self.initial_layer = nn.Linear(input_dim + protected_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout = nn.Dropout(p=0.2)  # Assuming keepRate of 0.8

    def forward(self, x, y_bool, z_masks):
        # Assuming y_bool is already concatenated to x if needed
        # z_masks is a list of boolean masks for each protected attribute class
        # In PyTorch, handling different masks might require adjusting input data before or during forward pass
        x = torch.cat([x] + z_masks, dim=1)  # Concatenating masks directly, adjust based on actual use case
        x = F.leaky_relu(self.initial_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))
            x = self.dropout(x)
        return torch.sigmoid(self.output_layer(x))  # Assuming binary classification

# Example instantiation and usage
input_dim = 10  # Example input feature dimension
protected_dim = 4  # Number of protected attribute classes
discriminator = Discriminator(input_dim, protected_dim)

# Example data
x_real = torch.randn(64, input_dim)  # Batch size of 64
y_real = torch.randint(0, 2, (64, 1)).float()  # Binary outcomes
z_masks = [torch.randint(0, 2, (64, 1)).float() for _ in range(protected_dim)]  # Protected attribute masks

# Forward pass
predictions = discriminator(x_real, y_real, z_masks)
print("Discriminator predictions shape:", predictions.shape)

def print2file(buf, outFile):
    """
    Appends a given string to a log file.

    Parameters:
    - buf (str): The data to write to the file.
    - outFile (str): The file path to append the data to.
    """
    with open(outFile, 'a') as file:  # Open the file in append mode
        file.write(buf + '\n')  # Write the buffer with a newline character

# Example usage:
buf = "Epoch 1: Loss=0.45, Accuracy=85%"
outFile = "/Users/penghuizhu/Desktop/Workspace/FairGAN/ctgan/GeneratedData/test.txt"
print2file(buf, outFile)

# This will append "Epoch 1: Loss=0.45, Accuracy=85%" to the file at /path/to/log.txt.
def generateData(nSamples=100, modelFile='model.pth', batchSize=100, outFile='out.npy', p_z=[],
                 p_y=[]):
    """
    Generates less-biased data using trained model and save to output path specified.

    Parameters:
    - nSamples (int): Size of entire original dataset.
    - modelFile (str): Path to trained Fair Transformer GAN model.
    - batchSize (int): Size of each batch.
    - outFile (str): Path to generated data files in numpy format.
    - p_z (list): Probability distribution of protected attribute.
    - p_y (list): Probability distribution of outcome.
    """

    # Load the trained model
    model = torch.load(modelFile)
    model.eval()  # Set the model to evaluation mode

    generated_data = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for _ in range(0, nSamples, batchSize):
            # Generate random inputs based on the distributions, if specified
            z = torch.tensor(np.random.choice(p_z, size=(batchSize,)),
                             dtype=torch.float32) if p_z else torch.randn(batchSize,
                                                                          model.randomDim)
            y = torch.tensor(np.random.choice(p_y, size=(batchSize,)),
                             dtype=torch.float32) if p_y else torch.randn(batchSize,
                                                                          model.randomDim)

            # Generate data
            fake_data = model.generate(z, y)  # Adjust this method call based on your model's generate method signature

            # Collect generated data
            generated_data.append(fake_data.cpu().numpy())

    # Concatenate and save generated data
    generated_data_np = np.concatenate(generated_data, axis=0)
    np.save(outFile, generated_data_np[:nSamples])  # Save only nSamples items


# Example usage
generateData(nSamples=1000, modelFile='model.pth', batchSize=100,
             outFile='generated_data.npy')

def calculateDiscAuc(preds_real, preds_fake):
    """
    Calculates discriminator AUC from real and fake predictions.
    """
    # Combine the predictions
    preds = np.concatenate([preds_real, preds_fake])
    # Create labels: 1 for real, 0 for fake
    labels = np.concatenate([np.ones_like(preds_real), np.zeros_like(preds_fake)])
    # Calculate AUC
    auc = roc_auc_score(labels, preds)
    return auc

def calculateDiscAccuracy(preds_real, preds_fake):
    """
    Calculates discriminator accuracy from real and fake predictions.
    """
    # Threshold predictions at 0.5
    preds_binary = np.concatenate([preds_real, preds_fake]) > 0.5
    labels = np.concatenate([np.ones_like(preds_real), np.zeros_like(preds_fake)])
    # Calculate accuracy
    acc = accuracy_score(labels, preds_binary)
    return acc

def calculateGenAccuracy(preds_real, preds_fake):
    """
    Calculates generator accuracy from real and fake predictions.
    The generator is considered accurate if the discriminator classifies
    its outputs (fake data) as real.
    """
    # For generator accuracy, we only consider the fake predictions
    # and how often the discriminator is fooled (i.e., predicts real for fake inputs)
    preds_binary = preds_fake > 0.5
    labels = np.ones_like(preds_fake)  # Generator aims for these to be classified as real
    acc = accuracy_score(labels, preds_binary)
    return acc

# Example usage
# preds_real and preds_fake should be numpy arrays containing the discriminator's
# confidence scores for real and fake data, respectively.
preds_real = np.random.uniform(low=0.8, high=1.0, size=100)  # Simulated predictions for real data
preds_fake = np.random.uniform(low=0.0, high=0.2, size=100)  # Simulated predictions for fake data

auc = calculateDiscAuc(preds_real, preds_fake)
disc_acc = calculateDiscAccuracy(preds_real, preds_fake)
gen_acc = calculateGenAccuracy(preds_real, preds_fake)

print(f"Discriminator AUC: {auc}")
print(f"Discriminator Accuracy: {disc_acc}")
print(f"Generator Accuracy: {gen_acc}")

def pair_rd(y_real, z_real):
    """
    Helper function to calculate total pairwise risk difference across all z protected attribute classes.
    """
    unique_classes = torch.unique(z_real)
    rd_sum = 0
    for i in unique_classes:
        for j in unique_classes:
            if i != j:
                y_i = y_real[z_real == i]
                y_j = y_real[z_real == j]
                rd_sum += abs(y_i.mean() - y_j.mean())
    return rd_sum / (len(unique_classes) * (len(unique_classes) - 1))

def calculateRD(y_real, z_real):
    """
    Calculates risk difference score across all z protected attribute classes during training.
    """
    risk_diff = pair_rd(y_real, z_real)
    return risk_diff

def calculateClassifierAccuracy(preds_real, y_real):
    """
    Calculates classifier accuracy between real y outcome and predicted y.
    """
    correct = (preds_real.round() == y_real).float()
    acc = correct.mean()
    return acc.item()

def calculateClassifierRD(preds_real, z_real, y_real):
    """
    Calculate classifier risk difference score across all z protected attribute classes during training.
    """
    rd = pair_rd(y_real, z_real)
    return rd

def create_z_masks(z_arr):
    """
    Create a z_mask for each class (max 5) of protected attribute in z array.
    """
    masks = []
    for val in torch.unique(z_arr):
        masks.append((z_arr == val).float())
    return masks


def train_fair_transformer_gan(
    dataPath,
    outPath,
    nEpochs=100,
    discriminatorTrainPeriod=1,
    generatorTrainPeriod=1,
    pretrainBatchSize=100,
    batchSize=100,
    pretrainEpochs=10,
    saveMaxKeep=5,
    p_z=None,
    p_y=None,
    modelPath=None
):
    # Ensure output path exists
    Path(outPath).mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(dataPath)
    X = data['X']  # Assuming 'X' key for features
    z = data['sex']  # Assuming 'z' key for protected attributes
    y = data['y']  # Assuming 'y' key for outcomes

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    z_tensor = torch.tensor(z, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create datasets and dataloaders
    dataset = TensorDataset(X_tensor, z_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

    # Initialize the model
    model = FairTransformerGAN(...)  # Add required arguments

    if modelPath:
        model.load_state_dict(torch.load(modelPath))

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizers
    optimizerD = Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(nEpochs):
        for i, (X_batch, z_batch, y_batch) in enumerate(dataloader):
            # Move data to device
            X_batch, z_batch, y_batch = X_batch.to(device), z_batch.to(device), y_batch.to(device)

            # Discriminator training
            for _ in range(discriminatorTrainPeriod):
                optimizerD.zero_grad()
                # Implement discriminator training steps
                # lossD.backward()
                optimizerD.step()

            # Generator training
            for _ in range(generatorTrainPeriod):
                optimizerG.zero_grad()
                # Implement generator training steps
                # lossG.backward()
                optimizerG.step()

        # Optionally save models periodically
        if epoch % saveMaxKeep == 0:
            torch.save(model.state_dict(), f"{outPath}/model_epoch_{epoch}.pth")

    # Save final model
    torch.save(model.state_dict(), f"{outPath}/model_final.pth")

    print("Training complete.")

# Example usage:
train_fair_transformer_gan(
    dataPath='/Users/penghuizhu/Desktop/Workspace/FairGAN/examples/csv/adult.csv',
    outPath='path/to/save/models/',
    nEpochs=100,
    discriminatorTrainPeriod=1,
    generatorTrainPeriod=1,
    pretrainBatchSize=100,
    batchSize=100,
    pretrainEpochs=10,
    saveMaxKeep=5
)
