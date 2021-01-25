# Apple Silicon Tensorflow Benchmark

## How to intall

* Install Xcode Command Line Tools
    ```bash
    xcode-select --install
    ```

* Install miniforge for arm64 (Apple Silicon) from [miniforge](https://github.com/conda-forge/miniforge)

* Download TensorFlow 2.4 from [Apple github](https://github.com/apple/tensorflow_macos). Extract and go under `arm64` directory:
    ```
    cd tensorflow_macos/arm64
    ```

* Create conda environment
    ```bash
    conda create --name tf24 python==3.8.6
    conda activate tf24
    ```
* Install the dependencies
    ```bash
    conda install -y pandas matplotlib scikit-learn jupyterlab

    # Install specific pip version and some other base packages
    pip install --force pip==20.2.4 wheel setuptools cached-property six

    # Install all the packages provided by Apple but TensorFlow
    pip install --upgrade --no-dependencies --force numpy-1.18.5-cp38-cp38-macosx_11_0_arm64.whl grpcio-1.33.2-cp38-cp38-macosx_11_0_arm64.whl h5py-2.10.0-cp38-cp38-macosx_11_0_arm64.whl tensorflow_addons-0.11.2+mlcompute-cp38-cp38-macosx_11_0_arm64.whl

    # Install additional packages
    pip install absl-py astunparse flatbuffers gast google_pasta keras_preprocessing opt_einsum protobuf tensorflow_estimator termcolor typing_extensions wrapt wheel tensorboard typeguard

    # Install TensorFlow
    pip install --upgrade --force --no-dependencies tensorflow_macos-0.1a1-cp38-cp38-macosx_11_0_arm64.whl
    ```

[Reference](https://towardsdatascience.com/tensorflow-2-4-on-apple-silicon-m1-installation-under-conda-environment-ba6de962b3b8)
