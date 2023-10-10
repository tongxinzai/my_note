#### 软连接
在Linux系统中，软链接（Symbolic Link）和硬链接（Hard Link）是文件系统中两种不同的链接方式。
软链接是一个指向目标文件或目录的特殊文件，类似于Windows系统中的快捷方式。软链接创建了一个新的文件，其中包含指向目标文件或目录的路径信息。软链接可以跨越文件系统边界，并且即使目标文件被删除，软链接仍然存在，但是软链接失效。软链接的权限与目标文件无关，并且可以指向不存在的文件。创建软链接可以使用`ln -s`命令。
硬链接是一个指向目标文件的物理链接，它们共享相同的inode和数据块。硬链接不会创建新的文件，而是使目标文件的硬链接数量增加。多个硬链接可以指向同一个文件，它们具有相同的内容和权限。如果原始文件被删除，硬链接仍然存在，因为它们共享相同的inode。硬链接不能跨越文件系统边界，而且只能链接到文件，不能链接到目录。创建硬链接可以使用`ln`命令。
下面是创建软链接和硬链接的示例：
创建软链接：
```
ln -s /path/to/target /path/to/link
```
创建硬链接：
```
ln /path/to/target /path/to/link
```
总结来说，软链接是指向目标文件的快捷方式，可以跨文件系统边界，但是当目标文件被删除时失效；而硬链接是目标文件的物理链接，共享相同的inode，不能跨文件系统边界，但即使目标文件被删除，硬链接仍然存在。
#### 查看ubuntu版本
要查看Ubuntu的版本，可以按照以下步骤进行操作：
1. 打开终端（Ctrl+Alt+T）。
2. 在终端中输入以下命令并按Enter键运行：
   ```
   lsb_release -a
   ```
   或者，你也可以尝试以下命令：
   ```
   cat /etc/os-release
   ```
3. 终端将显示包含Ubuntu版本信息的输出。在输出中，你可以看到Distributor ID、Description、Release和Codename等详细信息。
   例如：
   ```
   No LSB modules are available.
   Distributor ID: Ubuntu
   Description: Ubuntu 20.04.3 LTS
   Release: 20.04
   Codename: focal
   ```
   在上面的示例中，Ubuntu版本为20.04.3 LTS，发行代号为focal。
通过执行上述命令，你将能够轻松地查看Ubuntu版本信息。如果你有更多问题，请随时提问。
#### 查看当前端口号
在 Ubuntu 上，你可以使用以下方法来查看当前打开的端口号：
1. 使用命令行工具 netstat：打开终端，输入以下命令来查看当前打开的端口号：
```
netstat -tuln
```
该命令将显示当前打开的 TCP 和 UDP 端口号。
2. 使用命令行工具 ss：打开终端，输入以下命令来查看当前打开的端口号：
```
ss -tuln
```
该命令也将显示当前打开的 TCP 和 UDP 端口号。
3. 使用命令行工具 lsof：打开终端，输入以下命令来查看当前打开的端口号：
```
sudo lsof -i -P -n | grep LISTEN
```
该命令将显示正在监听的进程和对应的端口号。
注意，为了查看仅限于 root 用户的端口，请使用 sudo 运行上述命令。
这些方法将帮助你在 Ubuntu 上查看当前打开的端口号。端口号是应用程序监听的特定端口，用于网络通信。
#### 查看公网ip
要查看本地网络的公网 IP 地址，你可以使用以下方法：
1. 使用网站或在线工具：访问一个提供公网 IP 查询的网站或在线工具，例如 "what is my IP" 或 "my public IP"，在网页上会显示你的公网 IP 地址。
2. 使用命令行工具：打开终端或命令提示符窗口，然后执行以下命令：
   - 在 Windows 上，使用 `nslookup` 命令：
     ```
     nslookup myip.opendns.com resolver1.opendns.com
     ```
   - 在 macOS 或 Linux 上，使用 `curl` 命令：
     ```
     curl ifconfig.me
     ```
   - 在 Linux 上，使用 `wget` 命令：
     ```
     wget -qO- ifconfig.me
     ```
这些命令将从公网服务器获取你的公网 IP 地址，并显示在命令行或终端窗口中。
请注意，这些方法都依赖于你能够连接到互联网。如果你在使用路由器或防火墙等设备的局域网中，你将看到该网络的公网 IP 地址，而不是特定设备的公网 IP 地址。
#### 查看ssh服务状态
要查看Ubuntu上SSH服务的状态，你可以执行以下步骤：
1. 打开终端应用程序。
2. 在终端中，输入以下命令以检查SSH服务的状态：
   ```
   sudo systemctl status ssh
   ```
   这将显示SSH服务的当前状态信息，包括是否正在运行以及其他相关信息。
3. 如果SSH服务正在运行，你将看到类似以下内容的输出：
   ```
   ● ssh.service - OpenBSD Secure Shell server
      Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: enabled)
      Active: active (running) since Mon 2022-01-10 08:45:22 PST; 1 day 10h ago
      Main PID: 1234 (sshd)
      ...
   ```
   这表明SSH服务正在运行，并显示了其进程ID（PID）以及其他详细信息。
4. 如果SSH服务没有运行，你将看到类似以下内容的输出：
   ```
   ● ssh.service - OpenBSD Secure Shell server
      Loaded: loaded (/lib/systemd/system/ssh.service; disabled; vendor preset: enabled)
      Active: inactive (dead)
      ...
   ```
   这表明SSH服务没有运行。
#### 创建新用户
在Ubuntu中，您可以使用以下步骤创建新用户：
1. 打开终端：您可以使用快捷键Ctrl + Alt + T来打开终端，或者在菜单中搜索并打开"终端"应用程序。
2. 运行以下命令以使用root权限登录：
   ```
   sudo -i
   ```
   系统会要求您输入管理员密码。
3. 运行以下命令以创建新用户。将"newuser"替换为您想要使用的实际用户名：
   ```
   adduser newuser
   ```
   系统会提示您输入新用户的密码和一些其他信息，请按照提示完成。
4. 如果您希望新用户具有管理员权限（即sudo访问权限），您可以将其添加到sudo用户组。运行以下命令将新用户添加到sudo用户组：
   ```
   usermod -aG sudo newuser
   ```
5. 完成后，您可以使用以下命令切换到新用户：
   ```
   su - newuser
   ```
   系统会要求您输入新用户的密码。
现在，您已成功在Ubuntu中创建了一个新用户。新用户将具有自己的主目录和访问权限，并且可以使用其自己的用户名和密码登录系统。
#### 用scp传输文件或文件夹
使用SCP命令可以在Ubuntu中传输文件和文件夹。以下是示例命令，演示如何使用SCP进行传输：
传输文件：
```
scp /path/to/local/file username@remote_host:/path/to/remote/directory
```
- 将`/path/to/local/file`替换为您要传输的本地文件的路径和文件名。
- 将`username`替换为远程服务器的用户名。
- 将`remote_host`替换为远程服务器的IP地址或域名。
- 将`/path/to/remote/directory`替换为远程服务器上您想要将文件传输到的目录。
传输文件夹：
```
scp -r /path/to/local/folder username@remote_host:/path/to/remote/directory
```
- 将`/path/to/local/folder`替换为您要传输的本地文件夹的路径。
- 将`username`替换为远程服务器的用户名。
- 将`remote_host`替换为远程服务器的IP地址或域名。
- 将`/path/to/remote/directory`替换为远程服务器上您想要将文件夹传输到的目录。
运行上述命令后，系统会要求您输入远程服务器的密码。输入密码后，SCP将会将文件或文件夹传输到远程服务器上的指定目录。
请确保您具有适当的权限来访问本地和远程目录，并在命令中替换正确的路径和文件名。
#### 查看架构
在Ubuntu上查看架构，可以按照以下步骤进行操作：
1. 打开终端（Ctrl+Alt+T）。
2. 在终端中输入以下命令并按Enter键运行：
   ```
   uname -m
   ```
   这个命令将返回你当前系统的架构信息。
   例如，如果返回的结果是`x86_64`，则表示你的系统是64位架构。
   如果返回的结果是`i386`或`i686`，则表示你的系统是32位架构。
   如果返回的结果是`armv7l`，则表示你的系统是ARM架构。
#### 解压文件
在Ubuntu上在Ubuntu上解压文件，你可以使用以下几种常见的方法：
1. 使用命令行解压：
   - 解压`.zip`文件：
     ```
     unzip file.zip
     ```
   - 解压`.tar.gz`或`.tgz`文件：
     ```
     tar -zxvf file.tar.gz
     ```
   - 解压`.tar.bz2`或`.tbz2`文件：
     ```
     tar -jxvf file.tar.bz2
     ```
   - 解压`.tar.xz`文件：
     ```
     tar -Jxvf file.tar.xz
     ```
   - 解压`.rar`文件（需要安装rar软件包）：
     ```
     unrar x file.rar
     ```
#### 查看自己的计算机型号
在 Ubuntu 上，你可以使用以下命令来查看自己的计算机型号：
```bash
sudo dmidecode -s system-product-name
```
在终端中运行此命令时，将提示你输入密码。输入密码后，该命令将输出你计算机的型号信息。
另外，你还可以使用以下命令来获取更多有关系统的详细信息：
```bash
sudo dmidecode
```
这将显示一个包含有关计算机硬件和固件的详细信息的输出。你可以通过滚动查看输出，找到与系统型号相关的信息。
请注意，`dmidecode` 命令需要以管理员权限运行，因此可能需要输入密码来执行这些命令。
#### 设置清华源加速pip安装
设置清华源，加速pip安装：
```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
#### 挂载u盘
在Ubuntu上，可以通过以下步骤来挂载U盘：
1. 插入U盘：将U盘插入计算机的USB接口。
2. 打开终端：按下Ctrl + Alt + T键打开终端。
3. 查看设备列表：在终端中运行以下命令，查看系统中的设备列表和它们的挂载点：
4. 
   ```
   sudo fdisk -l
   或
   lsblk
   ```
   这将列出计算机上的所有设备，包括硬盘、分区和可移动设备（如U盘）。
5. 确定U盘设备：根据U盘的大小和设备名称，确定U盘所对应的设备。通常，U盘的设备名称为`/dev/sdX`（X为字母，表示不同的设备）。
6. 创建挂载点：在终端中运行以下命令，创建一个用于挂载U盘的目录：
   ```
   sudo mkdir /mnt/usb
   ```
   这将在`/mnt`目录下创建一个名为`usb`的目录作为挂载点。
7. 挂载U盘：在终端中运行以下命令，将U盘挂载到刚刚创建的挂载点上：
   ```
   sudo mount /dev/sdX /mnt/usb
   ```
   请将`/dev/sdX`替换为您U盘的设备名称。
8. 访问U盘：现在，您可以通过文件管理器或终端访问挂载的U盘。在文件管理器中，导航到`/mnt/usb`目录即可查看U盘的内容。
9. 卸载U盘：在您完成使用U盘后，可以通过以下命令卸载U盘：
   ```
   sudo umount /mnt/usb
   ```
   这将卸载U盘并使其从系统中断开连接。
#### 查看硬盘大小
在Ubuntu上，可以使用以下方法来查看当前磁盘的大小：
1. 打开终端：按下`Ctrl + Alt + T`键打开终端。
2. 运行df命令：在终端中运行`df -h`命令，如下所示：
   ```
   df -h
   ```
   这将显示所有已挂载的磁盘分区的详细信息，包括文件系统、挂载点、总大小、已用空间和可用空间等。
   如果您只想查看特定的磁盘分区，可以指定该分区的挂载点。例如，要查看`/dev/sda1`分区的大小，可以运行以下命令：
   ```
   df -h /dev/sda1
   ```
   `-h`选项用于以人类可读的格式显示磁盘大小，以便更容易理解。
3. 查看磁盘使用情况：在df命令的输出中，可以查看每个磁盘分区的总大小、已用空间、可用空间和使用百分比等信息。
请注意，在运行df命令时，可能需要一些时间来收集和计算磁盘使用情况。如果您的系统中有大量的磁盘分区或文件，可能需要更长的时间来完成操作。
另外，如果您想以图形化的方式查看磁盘使用情况，可以使用系统监视器（System Monitor）或其他磁盘使用情况监控工具，如 Disks等。这些工具通常提供更直观和可视化的界面来显示磁盘大小和使用情况。
#### 查看文件大小
在Ubuntu上，可以使用以下方法来查看文件的大小：
1. 打开终端：按下`Ctrl + Alt + T`键打开终端。
2. 使用`ls`命令查看文件大小：在终端中运行以下命令，可以查看当前目录下的文件及其大小：
   ```
   ls -lh
   ```
   `-l`选项用于以长格式显示文件信息，包括文件大小和权限等。`-h`选项用于以人类可读的方式显示文件大小。
   如果要查看特定文件的大小，可以在命令后面加上文件名。例如，要查看`file.txt`文件的大小，可以运行以下命令：
   ```
   ls -lh file.txt
   ```
3. 使用`du`命令查看文件/文件夹大小：`du`命令用于估算文件或文件夹的磁盘使用量。在终端中运行以下命令，可以查看指定文件或文件夹的大小：
   ```
   du -sh /path/to/file_or_folder
   ```
   `-s`选项用于仅显示总大小，`-h`选项用于以人类可读的方式显示文件/文件夹大小。请将`/path/to/file_or_folder`替换为您要查看的文件或文件夹的路径。
请注意，使用`du`命令查看文件夹大小时，它会递归地计算文件夹及其所有子文件夹和文件的总大小。
#### 复制文件夹
在Ubuntu上，要复制整个文件夹及其内容，可以使用`cp`命令的`-r`选项（递归复制）。以下是复制文件夹的步骤：
1. 打开终端：按下Ctrl + Alt + T键打开终端。
2. 使用cp命令进行复制：在终端中运行以下命令，将整个文件夹及其内容复制到目标位置。例如，要将`/path/to/source_directory`文件夹复制到`/path/to/destination_directory`目录下，可以运行以下命令：
   ```
   cp -r /path/to/source_directory /path/to/destination_directory
   ```
   `-r`选项用于递归复制，确保复制整个文件夹及其所有子文件夹和文件。
3. 确认复制结果：在执行复制命令后，可以通过进入目标位置并检查复制的文件夹及其内容是否存在来确认复制结果。
请注意，如果目标位置已经存在同名的文件夹，`cp`命令将会覆盖目标位置上的同名文件夹及其内容。如果您希望在发生冲突时进行确认，可以使用`-i`选项来进行交互式复制。例如：
```
cp -ri /path/to/source_directory /path/to/destination_directory
```
这将在复制文件夹时提示是否覆盖目标位置上的同名文件夹及其内容。
如果您希望在复制过程中显示详细输出，可以使用`-v`选项。例如：
```
cp -rv /path/to/source_directory /path/to/destination_directory
```
这将显示每个复制的文件和文件夹的详细信息。
请注意，对于某些系统文件夹和文件，可能需要管理员权限来进行复制操作。在这种情况下，您可以在`cp`命令前加上`sudo`来以管理员身份运行命令。例如：
```
sudo cp -r /path/to/source_directory /path/to/destination_directory
```
这将要求您输入管理员密码来执行复制操作。
#### 常用命令
**一般操作**
- **pwd（present working directory）**
显示当前的工作目录/路径。
  
- **cd (change directory)**
改变目录，用于输入需要前往的路径/目录。
有一些特殊命令也很常用 :
```text
前往同一级的另一个目录
cd ../directory name
cd .. 表示进入上层目录
cd ../.. 进入上上层目录，后面还可以加更多。
前往同一级的另一个目录
cd ../directory name
cd -  //表示返回上一次的目录
cd ~  //进入home主目录，即/home/用户名的简写
```
- **ls (list)**
ls 显示当前目录下的文件（不包括隐藏文件和缓存文件等）；
列出目录下所有文件
```text
ls -a 
```
ll , 以列表形式显示当前路径下的所有文件的详细信息（包括隐藏文件和缓存文件等）。
  
- **mkdir (make directory)**
创建目录，后面接上directory的名字。
```text
mkdir I_dont_care //创建一个“我不在乎”目录
```
- **rm (remove)**
删除文件，后面接上要删除的文件名。如果要删除目录，需要这样写：\-r表示向下递归删除  
\-f表示直接强制删除，没有任何提示  对于文件夹的删除一般用rm -rf  （文件夹删除必须有r，递归删除）  对于文件的删除一般用rm -f  （其实rm本身就可以完成文件删除，但用f更高效）
1.强制删除文件夹并提示
```
sudo rm -r 文件名```

例如:
```
sudo rm -r  /usr/local/include/opencv```
2.强制删除文件夹并不提示
```cpp
sudo rm -rf 文件名
```
3.删除文件
```cpp
sudo rm -f 文件名
```
比如usr中的文件夹无法手动删除，就需要用到命令行删除，但是需要注意使用rm命令删除的文件和文件夹不会出现在回收站，因此会永久删除，使用时需要谨慎。```

- **touch**
创建任意格式的文件，包括源代码、文本等等，通过后缀来决定。例如，.cpp/.cc是c++源代码，而.py是python源代码。
```text
touch hello_world.cpp  //创建hello_world源代码
```
  
- **cp (copy)**
复制命令。通用格式为
```text
cp -? <源文件/源目录> <目的目录>  //第一个"-?"表示参数，出发地在左，目的地在右
```
特别的，如果想把某目录下所有文件都复制，可以使用参数-r
```text
cp -r cangjingkong/ xuexi    //将canjingkong目录下的所有资源都复制到xuexi目录中
```
- **mv (move)**
移动+重命名命令。格式类似于cp命令
```text
mv -? <源文件/源目录> <目的目录> //第一个"-?"表示参数，出发地在左，目的地在右
```
以移动txt文件为例 可以分为以下三种情况：
```text
mv a.txt b.txt                 //出发地和目的地是同一路径，名称从a.txt变为b.txt，那仅仅是重命名
mv ~/目录1/a.txt ~/目录2       //出发地和目的地是不同路径，没有指定新的名称，那仅仅是移动
mv ~/目录1/a.txt ~/目录2/b.txt //出发地和目的地是不同路径，指定了新的名称，那就是移动+重命名
```
常用的例子有，
移动目录到另一目录中
```text
mv 目录1/ 目录2
```
将某目录下所有的文件和目录都移动到当前目录下
```text
mv ~/videos/p_hub .
```
  
- **gedit**
在桌面临时新建一个text editor（文本编辑器）显示文件内的文本，并且支持修改。按ctrl+c退出文件显示。
```text
gedit <文件名>
```
例如，
```abap
gedit single_ladies_contacts.csv
```
- **cat**
在终端打印出文本内容。
```text
cat <文件名>  //在terminal内部打印，和gedit相区分
```
  
- **code/nano/vi/vim**
使用Visual Studio Code/Nano/vi/vim这四种编辑器，打开或者新建一个源代码文件。
- **apt/apt-get**
更推荐使用apt命令而不是apt-get命令，它的命令更精简而且易用。
```text
sudo apt install <软件名>  //安装软件最简单的方式
sudo apt list               //查看所有已安装的软件列表
sudo apt search <软件名>       //搜索某个软件
sudo apt remove <软件名>       //删除某个软件包
sudo apt purge <软件名>        //删除某个软件包以及配置文件，更彻底
```
还有我们最最常用的更新相关命令
```text
sudo apt update
sudo apt upgrade
```
- **dpkg (Debian package)**
包管理工具。
首先是下载功能。先在官网下载软件的deb格式安装包，然后cd到下载文件夹，打开terminal（终端）输入：
```text
dpkg -i <.deb后缀的软件名>  //i 表示 install
```
其次是卸载功能。和apt系列命令类似，也可以查看安装列表，搜索指定安装包和卸载。
```text
dpkg -r <包的名字>  //r 表示 remove, 此种方法会保留配置文件
dpkg -P <包的名字>  //直接全删了，配置也不会保留
dpkg -l            //查看安装列表
dpkg -S <包的名字>   //搜索某个包
```
- **kill**
结束指定进程时使用，就比如某个软件不响应了，这时候kill就相当于windows系统中的任务管理器中的“结束进程”按钮。我们只要指定进程的编号（ID#)
```text
kill <ID#>  //结束编号为<ID#>的进程
```
进程编号如何获得？引出下一个函数。
  
- **ps (process status)**
查看所有进程；
```text
ps -A     
```
查看所有包含其他使用者的进程；
```text
ps -aux
```
关键字查找某个进程，这个办法用于结束指定进程很方便。
```text
ps -ef | grep <关键字>
```
  
- **grep**
Linux grep 命令用于查找文件里符合条件的字符串。
  
- **find**
用于查找目录中的文件。
  
- **ln (link files)**
插入链接。
```
ln -sft
ln -hard
```
- **chmod (change mode)**
改变权限。
```
chmod +x dir/file or. chmod 777 dir/file
```
改为可执行
  
- **du(disk usage)**
```
du -h -l -d 1
\-h: --human readable 会显示Mb, Kb, G之类的单位，方便阅读
\-d 1: 表示深度为1，只会查看下一级目录的空间占用大小
```
  
- df(disk space filesystem)
```
df -h
```
  
---
**基础但实用的操作**
如果碰到不会的命令，或者忘记了具体的options（操作选项），可以使用帮助命令：
```text
命令名 -h or --help
```
如果嫌每次都要sudo太麻烦，可以先登录，获取root权限。
```text
sudo su //输入并回车
//就会让你输入root密码
```
学会以下代码就可以在技能中写上“熟悉linux系统的开关机”
```text
reboot  //重启
poweroff //关机
```

#### bashrc 文件
 bashrc 文件是 Bash shell 的配置文件,用于设置shell的环境和默认参数。
当用户登录时,系统会读取该文件来配置 shell 环境,以便为用户提供一个自定义的工作环境。
bashrc 文件通常包含以下内容:
• 定义aliases,为命令设置别名,方便使用。例如 alias ll='ls -al' 为ls命令设置别名ll。
• 设置环境变量,如 PATH 变量添加需要运行的命令路径。
• 定义函数,可以自己定义一些命令函数。
• 设置命令提示符(PS1)的显示样式。
• 加载其他配置文件以扩展shell的功能。
• 其他个性化的shell配置。
bashrc 文件优点:
• 可以自定义shell环境,提高使用效率。
• 对所有shell实例(包括新打开的shell)都生效,提供一致的工作环境。
• 可以加载其他配置文件,扩展shell的功能。
• 可以设置别名以简化复杂的命令。
• 可以添加自己定义的函数实现简单的功能。
总之,bashrc 文件是 shell 用户自定义和优化环境的重要手段,可以最大限度地发挥 shell 的功能。
对于 Ubuntu 系统,bashrc 文件位于用户主目录下,即 /home/用户名/.bashrc。

#### profile文件
 .profile 文件是 Linux/Unix 系统中用户登录 shell 启动时读取的第一个配置文件。
它主要包含:
• 设置环境变量,如 PATH, LANG 等。这些环境变量对用户登录的所有 shell 实例都生效。
• 加载其他配置文件,如 .bashrc 等,以扩展 shell 的功能。
• 其他登录时需要设置的内容。
与 .bashrc 文件比较:.profile 文件有以下特点:
• 仅在用户登录时读取一次,对登录 shell 生效。而 .bashrc 每打开一个新 shell 都会读取。
• 通常用于设置环境变量等对整个登录会话都需要的内容。而 .bashrc 更注重对 shell 实例的配置。
• .profile 仅对 bash shell 生效,其他 shell 不读取该文件。而 .bashrc 只用于 bash shell,其他 shell 有对应的配置文件。
• .profile 可以加载 .bashrc 文件,但不 vice versa。.bashrc 不会读取 .profile。
对于 Ubuntu 系统, .profile 文件位于用户主目录下,即 /home/用户名/.profile。
一个典型的 .profile 文件内容如下:
```
# .profile
# Set environment variables
export PATH=$HOME/bin:$PATH
export LANG=en_US.UTF-8
# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi 
# User specific aliases and functions
if [ -f ~/.bashrc ]; then 
        . ~/.bashrc 
fi
```
从上面可以看出,该文件设置了 PATH 环境变量和 LANG 环境变量,并加载了 /etc/bashrc 和 ~/.bashrc 文件以扩展 shell 的功能。
总结:.profile 和 .bashrc 文件都非常重要,用于配置 shell 环境。理解了这两个文件作用及其区别,可以更好的定制自己的 shell 工作环境。

#### linux系统下查看本机所在局域网中所有设备IP


- 方法一 NMAP命令：
```
nmap –nsP 192.168.1.0/24 #从192.168.1.0到192.168.1.255所有IP
```

- NBTSCAN命令：
```
nbtscan 192.168.1.1-254  #查找出所有能ping通的IP并带其mac地址，本地的arp也有记录
```

#### 局域网的IP段
局域网的IP段通常是由路由器或网络管理员分配的，因此具体的IP段会因网络配置而异。然而，最常见的局域网IP段是私有IP地址范围，根据IPv4标准，私有IP地址范围如下：

- 10.0.0.0 到 10.255.255.255
- 172.16.0.0 到 172.31.255.255
- 192.168.0.0 到 192.168.255.255

这些IP段被保留用于局域网和内部网络。如果你是网络的管理员，你可以查看路由器或网络设备的配置来确定分配给局域网的确切IP段。如果你是普通用户，你可以尝试使用默认的私有IP地址范围来扫描局域网中的IP地址。