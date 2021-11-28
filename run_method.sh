export THEANO_FLAGS="device=cuda0,floatX=float32"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "Running $1"
conda activate py37 && python run_config.py $1
