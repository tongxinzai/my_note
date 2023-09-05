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