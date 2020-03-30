import os
import numpy as np
import crepe

# this data contains a sine sweep
file = os.path.join(os.path.dirname(__file__), 'sweep.wav')
f0_file = os.path.join(os.path.dirname(__file__), 'sweep.f0.csv')


def verify_f0():
    result = np.loadtxt(f0_file, delimiter=',', skiprows=1)

    # it should be confident enough about the presence of pitch in every frame
    assert np.mean(result[:, 2] > 0.5) > 0.98

    # the frequencies should be linear
    assert np.corrcoef(result[:, 1]) > 0.99

    os.remove(f0_file)


def test_sweep():
    crepe.process_file(file)
    verify_f0()


def test_sweep_cli():
    assert os.system("crepe {}".format(file)) == 0
    verify_f0()


def test_sweep_torch():
    crepe.process_file(file, backend='torch')
    verify_f0()


def test_activation_torch_tf():
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file)
    except ValueError:
        import sys
        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    *_, confidence_tf, activation_tf = crepe.predict(
        audio, sr, backend='tf')

    import torch
    audio = torch.as_tensor(audio).unsqueeze(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    audio = audio.to(device)
    *_, confidence_torch, activation_torch = crepe.predict(
        audio, sr, backend='torch')

    assert np.allclose(confidence_tf, confidence_torch)
    assert np.allclose(activation_tf, activation_torch)
