###分层分类简介

&emsp;&emsp;Linux操作系统分层分类模型根据分层系统架构的要求，以及各软件包之间的依赖关系给出操作系统的分层结构，并定义了各个层的功能范围。

&emsp;&emsp;在分层维度上，整个操作系统架构自下而上被分为内核层、核心层、系统层和应用层等四个层，分别被标注为L0，L1，L2，L3。每层分别定义了各自的功能范围，每一层均依赖于下层，并以接口形式仅向直接上层提供其功能。其中内核层作为此架构层次的最底层，不依赖于其它任何层，而是直接依赖于硬件接口（含处理器指令集、总线信号等），而应用层其上没有其它软件层。

&emsp;&emsp;在分类维度上，操作系统分层分类模型根据软件包所主要隶属的文件形态、软件包提供的功能和软件包的使用场景三个子维度进行分类。

&emsp;&emsp;在文件形态分类子维度上，每个软件包均可被划分为库、工具和服务三类中的其中一类，且任何一个软件包都属于且只属于一个形态分类。

在功能分类子维度上，每个软件包可能拥有一到多个功能。

&emsp;&emsp;在使用场景分类子维度上，根据软件包会被应用到的不同业务场景，将其场景信息作为软件包属性给予标记，每个软件包可以被标记一个或多个场景信息。

&emsp;&emsp;在整体操作系统分层分类模型中，每个软件包均属于某一层，且在分类上，属于某个形态，拥有一到多个功能，以及属于一到多个使用场景。每个软件包的上述分层分类信息作为软件包元数据的一部分，可以为操作系统的选型、维护、演进以及软件包管理提供指导依据。

分层分类模型框架 :

1.src  主目录存储 主模块代码 ，处理repo 源信息的代码

    fcfl.ini 存储相关文件的存储路径 ： 
    [storage] 数据库存储信息，用于处理分层分类的结果解读
    path=/var/fcfl/db/

    [DB] 数据库存储信息，用于存储分层分类的结果
    path=/var/fcfl/db/

    [repo] repo源信息，用于存储下载的repo库
    path=/var/fcfl/repofile/

    [csv] 包描述信息，包含包 description，summary，srcname，version，release 等信息
    path = /var/fcfl/csv/

    [dot] 包依赖关系
    path = /var/fcfl/dot/

    [layer]  暂时未用
    path = /var/fcfl/layer/

 下载完repo源内容后，可通过修改 src/fcfl.ini 文件 
 [repo]
 path= 新repo路径

2.algorithm 目录，包含四个文件夹 
    classer 分类算法
    Layer 分层算法
    func_classer 功能分类算法
    VectorLayer 矢量空间锚点计算法（目前未用）

    算法使用的数据配置文件为 data_config.py,其中 classer_save_path 字段用于指明类别训练模型存放的位置

模型采用 bert-base-uncased ，下载链接为 ：https://hf-mirror.com/bert-base-uncased ，根据模型下载后存放位置可修改代码 ：
src/algorithm/data_config.py 里 model_path字段的内容

