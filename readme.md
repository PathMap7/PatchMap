## PatchMap: Patch Placement Benchmark
This repository is the code implementation for paper 'Benchmarking Adversarial Patch Selection and Location'
### SETUP 
Pleasre run:
```
pip install -r requirements.txt
```
to make sure all requirements are installed.<br/>
Then, you will need to make sure that a directory named 'ImageNet' present:
```
PatchMap/
├── ImageNet/
│   ├── ILSVRC2012_img_val
│   ├── ILSVRC2012_devkit_t12_class
│   └── imagenet_bbox
```
### Running the benchmark
In order to run the benchmark run the following line:
```
python evaluate_benchmark.py
```
### Running ASR evaluation
In order to run ASR evaluation over the entire dataset, you will need to run
```
python test.py
```

### Footnote
This adversarial patches are from 'ImageNet-Patch: A Dataset for Benchmarking Machine Learning Robustness against Adversarial Patches'