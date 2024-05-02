import torch
import os
import torchaudio
import pandas as pd
import numpy as np

import diff_apf_pytorch.modules

from losses import MultiResolutionSTFTLoss
from losses import MSELoss
from dataset import TestAudioDataset
from model import ParameterNetwork

from evaluation_metrics import db_rms, dB_peak, thdn


def export_validation_file(epoch: int,
            dataloader: torch.utils.data.DataLoader,
            net: torch.nn.Module,
            equalizer: diff_apf_pytorch.modules.Processor,
            loss,
            log_dir: str = "logs",
            use_gpu: bool = False,
            sr: int = 44100):
    '''
    Export audio file with validation of the network using random audio files from the dataset.

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

def validate(epoch: int,
            dataloader: torch.utils.data.DataLoader,
            net: torch.nn.Module,
            equalizer: diff_apf_pytorch.modules.Processor,
            losses,
            loudness,
            quality,
            log_dir: str = "logs",
            use_gpu: bool = False,
            sr: int = 44100):
    '''
    Validate the network on random audio files from the dataset.

    Parameters:
        epoch (int): current epoch
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        net (torch.nn.Module): the trained network
        equalizer (diff_apf_pytorch.modules.Processor): the parametric all-pass filter
        losses: Loss functions
        log_dir (str): directory to save logs
        use_gpu (bool): use GPU for training
        sr (int): sample rate of the audio files
    '''

    # Evaluate the network
    net.eval()

    total_losses = {key: 0.0 for key in losses}
    total_loudness = {key: 0.0 for key in loudness}
    total_quality = {key: 0.0 for key in quality}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_x, target_y = data[0], data[1]

            # Move the input and target pairs to the GPU if available
            if use_gpu:
                input_x = input_x.cuda()
                target_y = target_y.cuda()

            # Predict the parameters of the all-pass filters using
            # the input signal as an input to the TCN
            p_hat = net(input_x, target_y)

            # Apply the estimated parameters to the input signal
            x_hat = equalizer.process_normalized(input_x, p_hat)

            # Convert the signals to the CPU
            x_hat = x_hat.squeeze(0).numpy()[0]
            target_y = target_y.squeeze(0).numpy()[0]

            mix = (target_y + x_hat) / 2

            # Calculate the loss
            for loudness_key, loudness_func in loudness.items():
                total_loudness[loudness_key] += loudness_func(mix)
            for quality_key, quality_func in quality.items():
                total_quality[quality_key] += quality_func(x_hat)
            
            total_samples += 1

        for loudness_key, loudness_val in total_loudness.items():
            print(loudness_key, loudness_val/total_samples)
        for quality_key, quality_val in total_quality.items():
            print(quality_key, quality_val/total_samples)


if __name__ == "__main__":
    # Set the random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # provide annotations to the SDDS dataset
    annotations = pd.read_csv("/home/hf1/Documents/soundfiles/annotations.csv")

    sample_rate = 44100
    audio_dir = "/home/hf1/Documents/soundfiles/SDDS_segmented_Allfiles"

    # Using the custom SDDS dataset which returns the input and target pairs
    test_dataset = TestAudioDataset(annotations, audio_dir=audio_dir, fs=sample_rate)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # Create the parametric all-pass instance
    equalizer = diff_apf_pytorch.modules.ParametricEQ(sample_rate, max_q_factor=1.0)

    # Create the loss function
    losses = {'MR-STFT Loss': MultiResolutionSTFTLoss(),
              'MSE Loss': MSELoss()}

    loudness = {'RMS': db_rms,
                'Peak': dB_peak}
    
    quality = {'THDN': thdn}

    # Create the parameter estimation network
    net = ParameterNetwork(equalizer.num_params)
    net.load_state_dict(torch.load("outputs/diff_apf/models/1TCN_diff_apf_epoch_10_loss_1.1950842142105103.pth",
                                   map_location=torch.device('cpu'))['model_state_dict'],)

    # export_validation_file(11, dataloader, net, equalizer, losses, use_gpu=False, sr=sample_rate)
    validate(11, dataloader, net, equalizer, losses, loudness, quality, use_gpu=False, sr=sample_rate)