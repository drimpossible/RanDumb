# RanDumb
 
This repository contains simplified code for the paper:

**RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning**  
[Ameya Prabhu*](https://drimpossible.github.io), [Shiven Sinha*](https://www.linkedin.com/in/shiven-sinha/), [Ponnurangam Kumaraguru](https://www.iiit.ac.in/people/faculty/PKguru/), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Ozan Sener+](https://ozansener.net/), [Puneet Dokania+](https://puneetkdokania.github.io)

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

Arguments you can freely tweak given a dataset and model: 
  - Use RanDumb or RanPAC embedding function (`--embed_mode`)
  - Embedding Dimension (`--embed_dim`)
  - Use Flip Augmentation (`--augment`)
  - Switch mahalanobis distance on/off (`--model SLDA` or `--model NCM`)
  - Switch embedding on/off (`--embed`) 

Additional details and default hyperparameters can be found in `parse_args` function in `src/main.py` 
  
### Results


##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares. 

**Notes:** Code given here is not online. Explanation below for why it does not vary from an online version. 
 
- **Computing covariance:** It is an exact online rank-1 update to compute empirical covariance. Online updates are very slow, but this calculation is exact!
- **Shrinkage estimation:** We want the shrinked covariance. Simple way is hyperparameter search over shrinkage parameters, just like other CL methods :)

This code reaches similar performance as my original (ugly, ginormous) implementation (within ±0.8%), which was entangled with 10s of traditional methods from which I chose this. They are not needed for reproducing RanDumb.

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
