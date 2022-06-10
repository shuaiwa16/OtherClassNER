# Introduction
We propose a novel model, Mining Undefined Classes from Other-class (MUCO), that can automatically induce different undefined classes from the other class to improve few-shot NER. With these extra-labeled undefined classes, our method will improve the discriminative ability of NER classifier and enhance the understanding of predefined classes with stand-by semantic knowledge. 

Our academic paper which describes MUCO in detail can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/ACL2021.pdf.

# How do I run the code?
```
1. sh run.sh
2. python3 get_proto.py
3. python3 get_binary_data.py
4. python3 train_binary_classifier.py
5. python3 concat_training_data.py
6. sh run.sh
```

# How do I cite？
For now, cite the ACL paper:
```
@article{tong2021learning,
  title={Learning from Miscellaneous Other-Class Words for Few-shot Named Entity Recognition},
  author={Tong, Meihan and Wang, Shuai and Xu, Bin and Cao, Yixin and Liu, Minghui and Hou, Lei and Li, Juanzi},
  journal={arXiv preprint arXiv:2106.15167},
  year={2021}
}
```
