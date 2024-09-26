
当前 模型框架 :
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

