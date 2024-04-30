import torch
import os
import torchaudio
import pandas as pd

import diff_apf_pytorch.modules

from losses import MultiResolutionSTFTLoss
from dataset import AudioDataset
from model import ParameterNetwork


def validate(epoch: int,
            dataloader: torch.utils.data.DataLoader,
            net: torch.nn.Module,
            equalizer: diff_apf_pytorch.modules.Processor,
            loss,
            log_dir: str = "logs",
            use_gpu: bool = False,
            sr: int = 44100):
    '''
    Validate the network on random audio files from the dataset.

    Parameters:
        epoch (int): current epoch
        dataset (AudioDataset): the dataset to validate on
        net (torch.nn.Module): the trained network
        equalizer (diff_apf_pytorch.modules.Processor): the parametric all-pass filter
        log_dir (str): directory to save logs
        use_gpu (bool): use GPU for training
        sr (int): sample rate of the audio files
    '''
    # Create the audio log directory
    audio_log_dir = os.path.join(log_dir, "audio")
    os.makedirs(audio_log_dir, exist_ok=True)

    # Evaluate the network
    net.eval()

    # Get a random audio pair from the dataset
    data_iter = iter(dataloader)
    data = next(data_iter)
    input_x, target_y = data[0], data[1]

    # Move the input and target pairs to the GPU if available
    if use_gpu:
        input_x = input_x.cuda()
        target_y = target_y.cuda()

    with torch.no_grad():
        # Predict the parameters of the all-pass filters using
        # the input signal as an input to the TCN
        p_hat = net(input_x, target_y)

        # Apply the estimated parameters to the input signal
        x_hat = equalizer.process_normalized(input_x, p_hat)

    # Plot the input, target, and predicted signals
    # plotting.plot_response(y.squeeze(0), x_hat, x, epoch=epoch)

    # Save the audio files as results
    target_filename = f"epoch={epoch:03d}_target.wav"
    input_filename = f"epoch={epoch:03d}_input.wav"
    pred_filename = f"epoch={epoch:03d}_pred.wav"
    target_filepath = os.path.join(audio_log_dir, target_filename)
    input_filepath = os.path.join(audio_log_dir, input_filename)
    pred_filepath = os.path.join(audio_log_dir, pred_filename)
    torchaudio.save(target_filepath, target_y.squeeze(0).cpu(), sr, backend="soundfile")
    torchaudio.save(input_filepath, input_x.squeeze(0).cpu(), sr, backend="soundfile")
    torchaudio.save(pred_filepath, x_hat.squeeze(0).cpu(), sr, backend="soundfile")

    print(loss(x_hat, target_y))

    import sounddevice as sd
    import time
    audio_before = (input_x.squeeze(0) + target_y.squeeze(0)).cpu().numpy() / 2
    audio_after = (x_hat.squeeze(0) + target_y.squeeze(0)).cpu().numpy() / 2

    sd.play(audio_before[0], sr)
    sd.wait()
    time.sleep(0.2)
    sd.play(audio_after[0], sr)
    sd.wait()


if __name__ == "__main__":
    # provide annotations to the SDDS dataset
    annotations = pd.read_csv("annotations.csv")

    sample_rate = 44100
    audio_dir = "C:/Users/hfret/Downloads/SDDS"

    # Using the custom SDDS dataset which returns the input and target pairs
    test_dataset = AudioDataset(annotations, audio_dir=audio_dir, fs=sample_rate)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # Create the parametric all-pass instance
    equalizer = diff_apf_pytorch.modules.ParametricEQ(sample_rate, max_q_factor=1.0)

    # Create the loss function
    loss = MultiResolutionSTFTLoss()

    # Create the parameter estimation network
    net = ParameterNetwork(equalizer.num_params)
    net.load_state_dict(torch.load("pretrained/diff_apf_epoch_10_loss_1.1950842142105103.pth",
                                   map_location=torch.device('cpu'))['model_state_dict'],)

    print("****** Training ******")
    validate(11, dataloader, net, equalizer, loss, use_gpu=False, sr=sample_rate)
    print("******** Done ********")