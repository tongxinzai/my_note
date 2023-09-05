 好的,这里是常用的Windows命令提示符(CMD)命令:
1. `dir` - 显示当前目录中的文件和子目录
2. `cd` - 更改当前目录
3. `md` 或 `mkdir` - 创建新目录
4. `rd` 或 `rmdir` - 删除目录
5. `copy` - 复制文件到指定位置
6. `move` - 移动/剪切文件到指定位置 
7. `ren` 或 `rename` - 重命名文件或目录
8. `del` 或 `erase` - 删除一个或多个文件
9. `cls` - 清空屏幕
10. `ping` - 测试网络连接
11. `ipconfig` - 显示IP地址配置
12. `netstat` - 显示网络连接、端口和Socket信息
13. `tracert` - 跟踪路由信息
14. `shutdown` - 关闭或重启计算机
15. `tasklist` - 显示正在运行的进程和任务
16. `taskkill` - 终止指定的进程或任务
17. `systeminfo` - 显示系统配置信息
18. `chkdsk` - 检查磁盘问题和错误
19. `format` - 格式化磁盘
20. `sfc` - 系统文件检查器,扫描系统文件并修复问题
### **关机、重启、注销、休眠、定时**
- 关机：`shutdown /s`
- 重启：`shutdown /r`
- 注销：`shutdown /l`
- 休眠：`shutdown /h /f`
- 取消关机：`shutdown /a`
- 定时关机：`shutdown /s /t 3600`（3600 秒后关机）
### **目录操作**
**切换目录，进入指定文件夹：**
- 切换磁盘：`d:`（进入 d 盘）
- 切换磁盘和目录：`cd /d d:/test`（进入 d 盘 test 文件夹）
- 进入文件夹：`cd \test1\test2`（进入 test2 文件夹）
- 返回根目录：`cd \`
- 回到上级目录：`cd ..`
- 新建文件夹：`md test`
**显示目录内容：**
- 显示目录中文件列表：`dir`
- 显示目录结构：`tree d:\test`（d 盘 test 目录）
- 显示当前目录位置：`cd`
- 显示指定磁盘的当前目录位置：`cd d:`
### **网络操作**
- 延迟和丢包率：`ping ip/域名`
- Ping 测试 5 次：`ping ip/域名 -n 5`
- 清除本地 DNS 缓存：`ipconfig /flushdns`
- 路由追踪：`tracert ip/域名`
### **进程/服务操作**
**进程管理：**
- 显示当前正在运行的进程：`tasklist`
- 运行程序或命令：`start 程序名`
- 结束进程，按名称：`taskkill /im notepad.exe`（关闭记事本）
- 结束进程，按 PID：`taskkill /pid 1234`（关闭 PID 为 1234 的进程）
**服务管理：**
- 显示当前正在运行的服务：`net start`
- 启动指定服务：`net start 服务名`
- 停止指定服务：`net stop 服务名`
## 保存为 .bat 可执行文件
我们可以将常用的命令输入记事本中，并保存为后缀为 `.bat` 的可执行文件。
以后只要双击该文件即可执行指定命令；将文件放入系统【启动】目录中，可以实现开机自动运行。

## 查看已安装的软件
要使用命令行（cmd）查看电脑中已安装的软件，可以使用以下命令：
1. 对于 Windows 操作系统，可以使用 `wmic` 命令：
```
wmic product get name
```
运行该命令后，将显示电脑中已安装的软件的列表。
2. 对于 Linux 操作系统，可以使用 `dpkg` 或 `apt` 命令：
```
dpkg --list
```
或
```
apt list --installed
```
3. 如果您使用的是 macOS 操作系统，可以尝试使用以下命令：
```
brew list
```
这将显示通过 Homebrew 安装的软件列表。
