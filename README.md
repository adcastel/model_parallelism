# model_parallelism

This simulator mimics de behavior of an environment for distributed DNN training when using the Model parallelism approach.
This code is (right now) under development (but not in the main task road) and there is not any release version. 
The dnn_full.c is still incomplete and pretend to read the models from a cvs file. It crashes in some hardware and we are fixing it (when we have time). 

alexnet.csv file is an exemple of the format needed for introducing a model.

You are wellcomed to participate in this code to make it full-operative.

If use it, please cite the work:

@inproceedings{castello2019analysis,
  
  title={Analysis of model parallelism for distributed neural networks},
  
  author={Castell{\'o}, Adri{\'a}n and Dolz, Manuel F and Quintana-Ort{\'\i}, Enrique S and Duato, Jos{\'e}},
  
  booktitle={Proceedings of the 26th European MPI Users' Group Meeting},
  
  pages={7},
  
  year={2019},
  
  organization={ACM}
  
}
