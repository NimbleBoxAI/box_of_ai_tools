### Original Project [Repository](https://github.com/MalongTech/research-charnet)

This is a modified readme of the original project
# Convolutional Character Networks

This project hosts the testing code for CharNet, described in our paper:

    Convolutional Character Networks
    Linjie Xing, Zhi Tian, Weilin Huang, and Matthew R. Scott;
    In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2019.

   
## Installation

```bash
conda create -n charnet python=3.7
conda activate charnet
pip3 install torch torchvision
cd into the charnet directory
pip3 install -r requirements.txt
python3 setup.py build develop
```


## Run
1. Please run `bash download_weights.sh` to download our trained weights. 
2. Put images to test in the input_dir and the results will be saved in the output_dir. 
3. then just run: 

    ```bash
    python3 tools/test.py
    ```
4. Runs on CPU by default.

## Citation

If you find this work useful for your research, please cite as:

    @inproceedings{xing2019charnet,
    title={Convolutional Character Networks},
    author={Xing, Linjie and Tian, Zhi and Huang, Weilin and Scott, Matthew R},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
    }
    
## Contact

For any questions, please feel free to reach: 
```
github@malongtech.com
```


## License

CharNet is CC-BY-NC 4.0 licensed, as found in the [LICENSE](LICENSE) file. It is released for academic research / non-commercial use only. If you wish to use for commercial purposes, please contact sales@malongtech.com.
