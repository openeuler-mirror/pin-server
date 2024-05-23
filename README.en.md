# pin-server

#### Description

Pin (Plug-IN framework) server provides plugin APIs for compiler optimization developers to develop optimization pass. Currently, only the statistical plugin demo display is supported. This project is continually being iterated, and developers are welcome to join.

#### Software Architecture

Pin (Plug-IN framework) provides ecological partners with compiler optimization development capabilities based on plug-in IR, allowing developers to use the plug-in API to develop compiler optimization passes, and provide optimization capabilities to the mainstream compilers (such as GCC) in the form of plugins. This framework uses the plug-in server component (Pin-Server) and the plug-in client module to form a proxy mode. Pin-Server provides developers with a plug-in API to develop compiler optimization passes.

#### Installation tutorial

```
$ mkdir build
$ cd build
$ cmake ../ -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm
$ make
```

#### Instructions for use

1. xxxx
2. xxxx
3. xxxx

#### Participate and contribute

1. Fork this warehouse
2. Create a new Feat_xxx branch
3. Submit code
4. New Pull Request

#### Gitee Feature

1. Use Readme_XXX.md to support different languages, such as Readme_en.md, Readme_zh.md
2. Gitee official blog blog.gitee.com
3. You can learn about excellent open source projects on Gitee at https://gitee.com/explore
4.  The full name of GVP is Gitee's most valuable open source project,  which is an excellent open source project evaluated comprehensively.
5. Official user manual provided by Gitee https://gitee.com/help
6. GGitee Cover Stars is a column used to showcase the style of Gitee members https://gitee.com/gitee-stars/