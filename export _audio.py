import torch
import torchaudio
from phase_difference_model import ParameterNetwork
import diff_apf_pytorch.modules
from dataset import phase_differece_feature


# Load the audio file
t_audio_path = "Drums-Snare Top-M80.wav"
i_audio_path = "Drums-Snare Bottom-M81.wav"

target_y, sample_rate = torchaudio.load(t_audio_path)
input_x, sample_rate = torchaudio.load(i_audio_path)

# resample audio to 44100
resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=44100)
target_y = resample(target_y)
input_x = resample(input_x)

# Load short clip
ct_audio_path = "SnareTop.wav"
ci_audio_path = "SnareBot.wav"

ctarget_y, sample_rate = torchaudio.load(ct_audio_path)
cinput_x, sample_rate = torchaudio.load(ci_audio_path)

# Take 0.5 seconds of the audio
ctarget_y = ctarget_y[:, :int(0.5 * sample_rate)]
cinput_x = cinput_x[:, :int(0.5 * sample_rate)]

# Load the pretrained model
# Create the parametric all-pass instance
equalizer = diff_apf_pytorch.modules.ParametricEQ(sample_rate, max_q_factor=1.0)
model = ParameterNetwork(equalizer.num_params)
model.load_state_dict(torch.load("outputs/MSE_PhaseFeature_diff_apf_2/models/MSE_PhaseFeature_epoch_40_loss_9.2026e-03.pth",
                                   map_location=torch.device('cpu'))['model_state_dict'],)
model.eval()

# Pass the audio through the model
# Normalize the input_x and target_y
cinput_x = cinput_x / cinput_x.abs().max()
ctarget_y = ctarget_y / ctarget_y.abs().max()

# Load the data through the dataloaders
cinput_x = cinput_x.unsqueeze(0)
ctarget_y = ctarget_y.unsqueeze(0)

input_x = input_x.unsqueeze(0)
target_y = target_y.unsqueeze(0)

feature = phase_differece_feature(cinput_x, ctarget_y)

with torch.no_grad():
    # Predict the parameters of the all-pass filters
    p_hat = model(feature)

    # Apply the estimated parameters to the input signal
    x_hat = equalizer.process_normalized(input_x, p_hat)

torchaudio.save("OUTPUT_AUDIO.wav", x_hat.squeeze(0).cpu(), sample_rate, backend="soundfile")