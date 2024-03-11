# RanDumb
 
This repository contains easier-to-run, simplified code for the paper:

**RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning**  

[[PDF](https://arxiv.org/abs/2402.08823)]
[[Slides]()]
[[Bibtex](https://github.com/drimpossible/RanDumb/#citation)]

## Libraries and Data Setup

* Install all requirements required to run the code on a Python 3.x by:
```	
# First, activate a new virtual environment
pip3 install -r requirements.txt [TBA]
# Get all the required datasets into the 'data/' folder
# Setup the feature files in the directory `feats/` 
python get_feats.py 
python get_feats_vit.py
```
 
## Usage

* To run the RanDumb model, you can simply specify conditions from arguments, an example command below:
```
$ python main.py --dataset cifar100 --model SLDA --augment --embed --embed_mode RanDumb --embed_dim 25000
```
Arguments you can freely tweak and default hyperparameters are given in the `parse_args` function in `main.py`
  
- Note that this code estimates the covariance matrix in one-go, as it is too slow to run the sample-by-sample SLDA.

### Results


##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares. 

**Notes:** Code given here is not online. Explanation below for why it does not vary from an online version. This code reaches similar performance as my original (ugly, ginormous) implementations of 100s of traditional methods which are not needed for reproducing RanDumb (±0.8%).
 
- **Computing covariance:** It is an exact online rank-1 update to compute empirical covariance. Online updates are very slow, but this calculation is exact!
- **Shrinkage estimation:** We want the shrinked covariance. Simple way is hyperparameter search over shrinkage parameters, just like other CL methods :)

## Citation

We hope RanDumb is valueable for your project! To cite our work:

```
@article{prabhu2024randumb,
  title={RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning},
  author={Prabhu, Ameya and Sinha, Shiven and Kumaraguru, Ponnurangam and Torr, Philip HS and Sener, Ozan and Dokania, Puneet K},
  journal={arXiv preprint arXiv:2402.08823},
  year={2024}
}
```
