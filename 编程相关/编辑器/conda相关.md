#### conda 常用指令
Conda 是一个流行的虚拟环境和包管理器，用于在 Python 环境中创建和管理不同的环境以及安装和管理包。下面是一些常用的 Conda 指令：
1. 创建虚拟环境：
   - `conda create --name <环境名称>`：创建一个新的虚拟环境。
   - `conda create --name <环境名称> python=<版本号>`：创建一个指定 Python 版本的虚拟环境。
2. 激活和退出虚拟环境：
   - `conda activate <环境名称>`：激活指定的虚拟环境。
   - `conda deactivate`：退出当前的虚拟环境。
3. 管理包：
   - `conda install <包名称>`：安装指定包。
   - `conda install <包名称>=<版本号>`：安装指定版本的包。
   - `conda update <包名称>`：更新指定包到最新版本。
   - `conda remove <包名称>`：移除指定包。
4. 列出环境和包信息：
   - `conda info --envs`：列出当前存在的虚拟环境。
   - `conda list`：列出当前虚拟环境中安装的包。
5. 其他常用指令：
   - `conda search <包名称>`：搜索可用的包。
   - `conda clean --all`：清理不再需要的缓存和未使用的包。
6. 删除环境：
   - conda remove --name <环境名称> --all
需要注意的是，上述指令是针对 Conda 命令行工具的常用操作，可以在终端或命令提示符中执行。此外，如果使用 Anaconda Navigator 图形界面工具，也可以通过界面来进行虚拟环境和包的管理。
如果你需要更详细的信息，可以查阅 Conda 的官方文档，该文档提供了更全面和详细的使用指南和命令说明。
#### 重命名
要重命名 Conda 环境，可以按照以下步骤进行操作：
1. 首先，确保当前没有激活任何环境。在终端或命令提示符中，运行以下命令来取消激活任何已激活的环境：
   ```
   conda deactivate
   ```
2. 然后，使用 `conda create` 命令创建一个新的环境。指定新的环境名称作为参数，例如 `<新环境名称>`：
   ```
   conda create --name <新环境名称> --clone <原环境名称>
   ```
   这将创建一个新的环境，并将原环境的所有软件包复制到新环境中。
3. 接下来，使用 `conda remove` 命令删除原环境。指定原环境名称作为参数，例如 `<原环境名称>`：
   ```
   conda remove --name <原环境名称> --all
   ```
   这将删除原环境及其所有软件包。
4. 最后，激活新环境。在终端或命令提示符中，运行以下命令激活新环境：
   ```
   conda activate <新环境名称>
   ```
现在，你已经成功地重命名了 Conda 环境。你可以使用新的环境名称继续使用和管理环境。请注意，在执行这些操作时要小心，并确保备份重要的环境和数据。

#### 安装conda
**1，[在清华镜像网站]([Index of /anaconda/miniconda/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/))上下载与系统对应的miniconda3安装的脚本文件**，需要先看一下自己的计算机架构是什么，否则安装会报错。
```bash
sudo wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh
```
**2，给`.sh`文件添加`x`执行权限**
```bash
sudo chmod u+x Miniconda3-py38_4.9.2-Linux-x86_64.sh
```
**3，运行脚本文件**
```bash
sudo ./Miniconda3-py38_4.9.2-Linux-x86_64.sh
```
**4，一直按`enter`，直到看完所有`License Agreement`的内容，输入 `yes` 接受条款**
**5，默认安装在`/root/miniconda3`下，也可以自定义安装路径，比如这里我自定义为自己的主目录（见红色框），回车**
**6，同意初始化**

#### root用户安装的anaconda如何给普通用户使用

服务器里安装了[anaconda](https://so.csdn.net/so/search?q=anaconda&spm=1001.2101.3001.7020)，结果发现只有root用户才能使用，其他的用户都无法使用conda命令！
【解决方案】
首先，root用户安装anaconda的时候，需要安装在普通用户可以访问的目录下，比如/usr/local、/opt、/home之类的
其次，普通用户登陆后，需要执行以下[conda](https://so.csdn.net/so/search?q=conda&spm=1001.2101.3001.7020) init 使conda的路径等系统环境变量信息写入当前用户的bashrc下
例：
anaconda装在了   
>  /home/conda/anaconda3

普通用户需要执行   
> /home/conda/anaconda3/bin/conda init bash

 你需要根据自己的实际情况修改bin之前的目录
 普通用户再执行启用配置命令
> source ~/.bashrc