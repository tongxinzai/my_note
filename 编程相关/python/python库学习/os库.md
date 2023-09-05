在Python的`os`模块中，提供了一些基本的操作对象用于处理与操作系统相关的功能。下面是一些常用的`os`模块中的操作对象：
1. `os.name`: 获取当前操作系统的名称，如`posix`（Linux和Unix系统）、`nt`（Windows系统）等。
2. `os.getcwd()`: 获取当前工作目录的路径。
3. `os.chdir(path)`: 改变当前工作目录到指定的路径。
4. `os.listdir(path)`: 返回指定目录下的文件和子目录列表。
5. `os.mkdir(path)`: 创建指定路径的目录。
6. `os.makedirs(path)`: 递归地创建多层目录。
7. `os.remove(path)`: 删除指定路径的文件。
8. `os.rmdir(path)`: 删除指定路径的空目录。
9. `os.removedirs(path)`: 递归地删除多层空目录。
10. `os.rename(src, dst)`: 将文件或目录从src重命名为dst。
11. `os.path.join(path1, path2, ...)`: 将多个路径组合成一个路径。
12. `os.path.abspath(path)`: 返回指定路径的绝对路径。
13. `os.path.exists(path)`: 判断指定路径是否存在。
14. `os.path.isdir(path)`: 判断指定路径是否为目录。
15. `os.path.isfile(path)`: 判断指定路径是否为文件。
16.  `os.environ`: 提供对操作系统环境变量的访问。
17. `os.system(command)`: 在子shell中执行系统命令。
18. `os.walk(top, topdown=True, onerror=None, followlinks=False)`: 遍历指定目录及其子目录中的文件。
19. `os.access(path, mode)`: 检验对指定路径的访问权限。
20. `os.utime(path, times)`: 修改指定路径的访问和修改时间。
这些操作对象涵盖了文件和目录的创建、删除、修改，以及执行系统命令等功能。它们可以帮助你进行更高级的操作，如运行外部程序、遍历文件系统等。