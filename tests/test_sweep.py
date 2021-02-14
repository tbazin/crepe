import os
import subprocess
import numpy as np
import crepe

# this data contains a sine sweep
file = os.path.join(os.path.dirname(__file__), "sweep.wav")
f0_file = os.path.join(os.path.dirname(__file__), "sweep.f0.csv")


def verify_f0():
    result = np.loadtxt(f0_file, delimiter=",", skiprows=1)

    # it should be confident enough about the presence of pitch in every frame
    assert np.mean(result[:, 2] > 0.5) > 0.98

    # the frequencies should be linear
    assert np.corrcoef(result[:, 1]) > 0.99

    os.remove(f0_file)


def test_sweep():
    crepe.process_file(file)
    verify_f0()


def test_sweep_cli():
    result = subprocess.run(["crepe", file])
    assert result == 0
    verify_f0()


def test_sweep_torch():
    crepe.process_file(file, backend="torch")
    verify_f0()


# test consistency of results between PyTorch and TF
# passes only if using very lax parameters for the np.allclose comparison,
# not sure if it's due to floating point numerical imprecisions
# or to an actual bug...
def test_activation_torch_tf():
    try:
        from scipy.io import wavfile

        sr, audio = wavfile.read(file)
    except ValueError:
        import sys

        print("CREPE: Could not read %s" % file, file=sys.stderr)
        raise

    *_, confidence_tf, activation_tf = crepe.predict(audio, sr, backend="tf")

    *_, confidence_torch, activation_torch = crepe.predict(audio, sr, backend="torch")

    from functools import partial

    relaxed_allclose = partial(np.allclose, rtol=1e-2, atol=1e-8)
    assert relaxed_allclose(confidence_tf, confidence_torch[0].cpu().numpy())
    assert relaxed_allclose(activation_tf, activation_torch[0].cpu().numpy())
