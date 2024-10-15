### Introduction pkgs-categorizer 

&emsp;&emsp;The pkgs-categorizer model of the Linux operating system provides a layered structure of the operating system based on the requirements of the hierarchical system architecture and the dependency relationships between various software packages, and defines the functional scope of each layer.

&emsp;&emsp;In terms of hierarchical dimensions, the entire operating system architecture is divided into four layers from bottom to top: the kernel layer, the core layer, the system layer, and the application layer, labeled as L0, L1, L2, and L3 respectively. Each layer defines its own functional scope, where each layer depends on the layer below it and provides its functions solely to the immediate upper layer in the form of interfaces. Among them, the kernel layer, as the lowest level of this hierarchical architecture, does not depend on any other layer but directly depends on hardware interfaces (including processor instruction sets, bus signals, etc.). The application layer, on the other hand, has no other software layer above it.

&emsp;&emsp;On the classification dimension, the hierarchical classification model of the operating system categorizes software packages based on three sub-dimensions: the primary file type they belong to, the functions they provide, and the usage scenarios in which they are employed.

&emsp;&emsp;On the sub-dimension of file type classification, each software package can be categorized into one of the three categories: libraries, tools, and services, and each software package belongs to and only belongs to one of these morphological classifications.On the sub-dimension of functional classification, each software package may possess one or multiple functions.

&emsp;&emsp; On the sub-dimension of usage scenario classification, based on the different business scenarios in which a software package will be applied, its scenario information is labeled as an attribute of the software package. Each software package can be tagged with one or multiple scenario information.

&emsp;&emsp;In the overall hierarchical classification model of the operating system, each software package belongs to a specific layer, and in terms of classification, it falls into a particular form, possesses one or more functions, and pertains to one or more usage scenarios. The aforementioned hierarchical classification information of each software package, as part of the software package metadata, can provide guidance for the selection, maintenance, evolution of the operating system, as well as for software package management.

pkgs-categorizer framework :

1.The src main directory stores the main module code and the code for handling repository source information.

    The fcfl.ini file stores the storage paths for related files:

    [storage]  
    # Database storage information for interpreting hierarchical classification results  
    path=/var/fcfl/db/  
  
    [DB]  
    # Database storage information for storing hierarchical classification results  
path=/var/fcfl/db/  
  
[repo]  
    # Repository source information for storing downloaded repositories  
    path=/var/fcfl/repofile/  
  
    [csv]  
    # Package description information, including package description, summary, srcname, version, release, etc.  
    path=/var/fcfl/csv/  
  
    [dot]  
    # Package dependency relationships  
    path=/var/fcfl/dot/  
  
    [layer]  
    # Currently unused  
    path=/var/fcfl/layer/
   
    After downloading the content of the repository source, you can modify the src/fcfl.ini file by setting the new repository path in the [repo] section:
    [repo]  
    path=新repo路径


2.The "algorithm" directory contains four folders:

    classer: Classification Algorithms
    Layer: Hierarchical Algorithms
    func_classer: Functional Classification Algorithms
    VectorLayer: Vector Space Anchor Point Calculation Algorithm (currently unused)
    
The data configuration file used by the algorithms is data_config.py, where the classer_save_path field specifies the location where the category training models are stored.

The model adopted is bert-base-uncased, and the download link is: https://hf-mirror.com/bert-base-uncased . Based on the location where the model is saved after downloading, you can modify the content of the model_path field in src/algorithm/data_config.py.

