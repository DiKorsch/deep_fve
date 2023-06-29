# DeepFVE for PyTorch

## Installation
```bash
conda install -c nvidia cupy cudatoolkit~=11.7.0 nccl cudnn
```



## Running tests:

```bash
python run_tests.py
```
### Additional requirements for the tests:

```bash
conda install -c conda-forge cyvlfeat
```


## Example usage
*TBD*

## License
This work is licensed under a [GNU Affero General Public License][agplv3].

[![AGPLv3][agplv3-image]][agplv3]

[agplv3]: https://www.gnu.org/licenses/agpl-3.0.html
[agplv3-image]: https://www.gnu.org/graphics/agplv3-88x31.png

## Citation
You are welcome to use our code in your research! If you do so please cite it as:

```bibtex
@inproceedings{Korsch21:ETE,
    author = {Dimitri Korsch and Paul Bodesheim and Joachim Denzler},
    booktitle = {German Conference on Pattern Recognition (DAGM-GCPR)},
    title = {End-to-end Learning of Fisher Vector Encodings for Part Features in Fine-grained Recognition},
    pages = {142--158},
    doi = {10.1007/978-3-030-92659-5_9},
    year = {2021},
}
```
