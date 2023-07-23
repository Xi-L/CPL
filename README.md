# CPL

Code for ICML2023 Paper: Continuation Path Learning for Homotopy Optimization

The code is mainly designed to be simple and readable, it contains:

- <code>run_nonconvex_opt.py</code> is a ~130-line script to run the Continuation Path Learning (CPL) algorithm for nonconvex optimization;
- <code>run_noisy_regression.py</code> is a ~70-line script to run the Continuation Path Learning (CPL) algorithm for noisy regression;
- <code>model.py</code> is a simple FC Continuation Path Model;
- <code>function.py</code> contains all the test problems used in the paper;


**Reference**

If you find our work is helpful to your research, please cite our paper:
```
@inproceedings{linpareto,
  title={Continuation Path Learning for Homotopy Optimization},
  author={Lin, Xi and Yang, Zhiyuan and Zhang, Xiaoyuan and Zhang, Qingfu},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
