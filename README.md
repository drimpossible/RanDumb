# RanDumb
 
This repository contains simplified code for the paper:

**RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning**  
[Ameya Prabhu*](https://drimpossible.github.io), [Shiven Sinha*](https://www.linkedin.com/in/shiven-sinha/), [Ponnurangam Kumaraguru](https://www.iiit.ac.in/people/faculty/PKguru/), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Ozan Sener+](https://ozansener.net/), [Puneet Dokania+](https://puneetkdokania.github.io)

[[PDF](https://arxiv.org/abs/2402.08823)]
[[Slides]()]
[[Bibtex](https://github.com/drimpossible/RanDumb/#citation)]

<p align="center">
<a href="url"><img src="https://github.com/drimpossible/GDumb/blob/main/Pull.png" height="300" width="381" ></a>
</p>

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
  
- Script to run RanDumb experiments given in `scripts/runall.sh`

**Notes:** Code given here is not online. Explanation below for why it does not vary from an online version. 
 
- **Computing covariance:** It is an exact rank-1 update to compute empirical covariance, so the matrix should be the same. (Also the online code is very slow to run)!
- **Shrinkage estimation:** We want the shrinked covariance. We use OAS method to obtain shrinkage parameters, but note that those parameters are a function of the empirical covariance matrix and again can be perfectly estimated online.
- **No lambda used when inverting:** Goal is to be hyperparameter optimization free (it is not nice to do in CL, can write a paper about gains from hparam optimization), we can make it hparam free by setting $lambda=0$ when inverting the covariance matrix-- no ridge regression style stuff here. This change leads to *slightly* worse results than original code used for the paper.

Overall, this code reaches similar performance (within ±0.8%) but is far more hackable-- my original (ugly, ginormous) implementation was entangled with my implementations of many many traditional kernels/online methods from which I chose this particular combination of kernel and classifier. 

#### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

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
