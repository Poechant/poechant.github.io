---
layout: post
title:  OpenRTMFP/Cumulus 原理及源码解读 5：IO 管理源码分析
date:   2012-04-24 11:31:10 +0800
categories: rt_tech
tags: [直播技术]
description: 
excerpt: 
---

**本文目录**
* TOC
{:toc}

### 一、流缓冲区

这段我们主要分析 MemoryStream.h 文件中定义的类。

#### 1、了解 `std::streambuf`

首先要了解 `streambuf` 内置了一个 `get` 指针和一个 `put` 指针。`streambuf` 的所有操作基本都是对这两个指针的操作。其一些成员函数的缩写中的 `g` 和 `p` 就分别表示 get pointer 和 put pointer。

##### 1.1、单步移动内置指针

Increase get pointer: Advances the get pointer by `n` positions. The get pointer is the internal pointer that points to the next location in the controlled input sequence.

```c++
  void gbump ( int n );
```

Increase put pointer: Advances the put pointer by `n` positions. The put pointer is the internal pointer that points to the next location of the controlled output sequence.

```c++
  void pbump ( int n );
```

##### 1.2、获取 get 指针和 put 指针的位置

Pointer to current position of input sequence: Returns a reference to the current element of the controlled input sequence (i.e., the “get pointer”).

```c++
  char * gptr ( ) const;
```

Pointer to current position of output sequence: Returns a reference to the current element of the output sequence (the put pointer).

```c++
  char * pptr ( ) const;
```

##### 1.3、设置 `get` 和 `put` 指针可达区域的上下界

Set input sequence pointers: Sets values for the pointers that define both the boundaries of the accessible part of the controlled input sequence and the get pointer itself.

```c++
  void setg ( char* gbeg, char* gnext, char* gend );
```

* `gbeg`: New value for the pointer to the beginning of the accessible part of the controlled input sequence.
gnext: New value for the get pointer, which points to the next element within the controlled input sequence where the next input operation shall be performed.
* `gend`: New value for the end pointer, just past the end of the accessible part of the controlled input sequence.
* Set output sequence pointers: Sets the values that define the boundaries of the accessible part of the controlled output sequence.

```c++
  void setp ( char* pbeg, char* pend );
```

* `pbeg`: New value for the pointer to the beginning of the accessible part of the controlled output sequenceand put pointer.
* `pend`: New value for the end pointer, just past the end of the accessible part of the controlled output sequence.

#### 2、`MemoryStreamBuf`

类定义：

```c++
class MemoryStreamBuf: public std::streambuf {
    friend class ScopedMemoryClip;
public:
    MemoryStreamBuf(char* pBuffer,Poco::UInt32 bufferSize);
    MemoryStreamBuf(MemoryStreamBuf&);
    ~MemoryStreamBuf();
 
    void            next(Poco::UInt32 size); // Explaint below
    Poco::UInt32    written(); // Explaint below
    void            written(Poco::UInt32 size);
    Poco::UInt32    size();  // Explaint below
    void            resize(Poco::UInt32 newSize); // Explaint below
    char*           begin(); // Explaint below
    void            position(Poco::UInt32 pos=0); // Explaint below
    char*           gCurrent(); // Explaint below
    char*           pCurrent(); // Explaint below
 
private:
    virtual int overflow(int_type c);
    virtual int underflow();
    virtual int sync();
 
    Poco::UInt32    _written;
    char*           _pBuffer;
    Poco::UInt32    _bufferSize;
 
    MemoryStreamBuf();
    MemoryStreamBuf& operator = (const MemoryStreamBuf&);
};
```

`ScopedMemoryClip` 是 `MemoryStreamBuf` 的友元，其内部有 `MemoryStreamBuf` 的成员，这里暂且不管。构造函数传入的参数是缓冲区的地址和缓冲区大小（字节数）。拷贝构造函数和析构函数不必赘述。

##### 2.1、移动内置的 `get` 和 `put` 指针：

`put` 和 `get` 指针都移动：

```c++
void MemoryStreamBuf::next(UInt32 size) {
    pbump(size);
    gbump(size);
}
```

##### 2.2、获取 get 和 put 指针当前位置：

封装 `streambuf` 的 `gptr` 和 `pptr`：

```c++
inline char* MemoryStreamBuf::gCurrent() {
    return gptr();
}
inline char* MemoryStreamBuf::pCurrent() {
    return pptr();
}
```

##### 2.3、获取缓冲区的起始位置和大小：

依赖于内置成员变量 pBuffer 和 bufferSize：

```c++
inline char* MemoryStreamBuf::begin() {
    return _pBuffer;
}
 
inline Poco::UInt32 MemoryStreamBuf::size() {
    return _bufferSize;
}
```

##### 2.4、缓冲区的已写字节数

读取（其中也可能发生设置操作）：

```c++
UInt32 MemoryStreamBuf::written() {
    int written = pCurrent() - begin(); // 已写字节数
    if (written < 0)
        written = 0;
    if (written > _written) // 保存已写字节数
        _written = (UInt32)written;
    return _written;
}
```

设置：

```c++
void MemoryStreamBuf::written(UInt32 size) {
    _written=size;
}
```

##### 2.5、显式设定 `put` 和 `get` 指针位置

设定 put 和 get 指针为以缓冲区首地址为开始偏移量为 pos 的位置：

```c++
void MemoryStreamBuf::position(UInt32 pos) {
 
    // 保存已写字节数
    written(); // Save nb char written
 
    // 移动 put 指针
    setp(_pBuffer, _pBuffer + _bufferSize);
    if (pos > _bufferSize)
        pos = _bufferSize;
    pbump((int) pos);
 
    // 移动 get 指针
    setg(_pBuffer, _pBuffer + pos, _pBuffer + _bufferSize);
}
```

##### 2.6 修改缓冲区大小

```c++
void MemoryStreamBuf::resize(UInt32 newSize) {
    // 大小标识
    _bufferSize = newSize;
 
    // gptr 当前位置
    int pos = gCurrent() - _pBuffer;
    if (pos > _bufferSize) pos = _bufferSize;
 
    // 设置 gptr 可达范围和当前位置
    setg(_pBuffer, _pBuffer + pos, _pBuffer + _bufferSize); 
    // pptr 当前位置
    pos = pCurrent() - _pBuffer;
    if (pos > _bufferSize) pos = _bufferSize;
 
    // 设置 pptr 可达范围和当前位置
    setp(_pBuffer,_pBuffer + _bufferSize);
        pbump(pos);
}
```

##### 2.7、构造函数、拷贝构造函数和析构函数

构造函数会设定 `pptr` 和 `gptr`，并初始化 `pBuffer` 和 `bufferSize`。

```c++
MemoryStreamBuf::MemoryStreamBuf(char* pBuffer, UInt32 bufferSize):     _pBuffer(pBuffer),_bufferSize(bufferSize),_written(0) {
    setg(_pBuffer, _pBuffer,_pBuffer + _bufferSize);
    setp(_pBuffer, _pBuffer + _bufferSize);
}
```

析构函数会拷贝对方的 `pBuffer`、`bufferSizse`、`_written`，并设定 `gptr`、`pptr`。注意设定 `pptr` 时，要分别调用 `setp` 和 `pbump`，因为 `setp` 仅将 `pptr` 设定为传入的首个参数值（与可达范围的首地址相同）。

```c++
MemoryStreamBuf::MemoryStreamBuf(MemoryStreamBuf& other):   _pBuffer(other._pBuffer),_bufferSize(other._bufferSize),_written(other._written) {
    setg(_pBuffer, other.gCurrent(), _pBuffer + _bufferSize);
    setp(_pBuffer, _pBuffer + _bufferSize);
    pbump((int)(other.pCurrent()-_pBuffer));
}
```

析构函数：

```c++
MemoryStreamBuf::~MemoryStreamBuf() {
}
```

### 二、IO 流

#### 1、了解 `std::ios`

Initialize object [`protected`]: This protected member initializes the values of the stream’s internal flags and member variables.

```c++
  void init ( streambuf* sb );
```

初始化后如下函数的返回值：

| member function | value                                                 |
|-----------------|-------------------------------------------------------|
| rdbuf()         | sb                                                    |
| tie()           | 0                                                     |
| rdstate()       | goodbit if sb is not a null pointer, badbit otherwise |
| exceptions()    | goodbit                                               |
| flags()         | skipws \| dec                                         |
| width()         | 0                                                     |
| precision()     | 6                                                     |
| fill()          | ‘ ’ (whitespace)                                      |
| getloc()        | a copy of locale()                                    |

#### 2、`MemoryIOS`

`MemoryIOS` 封装 `MemoryStreamBuf`，且是 `MemoryInputStream` 和 `MemoryOutputStream`的基类，用以确保流缓冲区和基类的初始化序列的正确性。该类继承自 `std::ios`。

```c++
class MemoryIOS: public virtual std::ios
{
public:
    MemoryIOS(char* pBuffer,Poco::UInt32 bufferSize);
    MemoryIOS(MemoryIOS&);
    ~MemoryIOS();
    MemoryStreamBuf* rdbuf();
    virtual char*   current()=0;
    void            reset(Poco::UInt32 newPos);
    void            resize(Poco::UInt32 newSize);
    char*           begin();
    void            next(Poco::UInt32 size);
    Poco::UInt32    available();
private:
    MemoryStreamBuf _buf;
};
```

##### 2.1、构造函数、拷贝构造函数和析构函数

```c++
MemoryIOS::MemoryIOS(char* pBuffer, UInt32 bufferSize):_buf(pBuffer, bufferSize) {
    poco_ios_init(&_buf);
}
```

`poco_ios_init` 为 `init` 的宏定义，用于初始化成员 `_buf`。

```c++
MemoryIOS::MemoryIOS(MemoryIOS& other):_buf(other._buf) {
    poco_ios_init(&_buf);
}
```

拷贝构造函数同构造函数。如下的析构函数不必赘述：

```c++
MemoryIOS::~MemoryIOS() {
}
```

##### 2.2、得到 `MemoryStreamBuf` 成员的地址

```c++
inline MemoryStreamBuf* MemoryIOS::rdbuf() {
    return &_buf;
}
```

##### 2.3、当前位置

这是一个纯虚函数，由 `MemoryInputStream` 和 `MemoryOutpuStream` 继承时实现：

```c++
virtual char*   current()=0;
```

##### 2.4、封装 `MemoryStreamBuf` 成员的一些函数

`begin`

```c++
  inline char* MemoryIOS::begin() {
      return rdbuf()->begin();
  }
```

`resize`

```c++
  inline void MemoryIOS::resize(Poco::UInt32 newSize) {
      rdbuf()->resize(newSize);
  }
```

`next`

```c++
  inline void MemoryIOS::next(Poco::UInt32 size) {
      rdbuf()->next(size);
  }
```

`position` 封装为 `reset`

```c++
  void MemoryIOS::reset(UInt32 newPos) {
      if(newPos>=0)
          rdbuf()->position(newPos);
      clear();
  }
```

##### 2.5 缓冲区可读数据的字节数

```c++
UInt32 MemoryIOS::available() {
    int result = rdbuf()->size() - (current() - begin()); // 缓冲区剩余可读数据字节数
    if (result < 0)
        return 0;
    return (UInt32)result;
}
```

#### 3、输入流

```c++
class MemoryInputStream: public MemoryIOS, public std::istream
{
public:
    MemoryInputStream(const char* pBuffer,Poco::UInt32 bufferSize);
        /// Creates a MemoryInputStream for the given memory area,
        /// ready for reading.
    MemoryInputStream(MemoryInputStream&);
    ~MemoryInputStream();
        /// Destroys the MemoryInputStream.
    char*           current();
};
```

构造函数、拷贝构造函数和析构函数也都没什么可说的，初始化 `MemoryIOS` 以及 `istream`。`istream` 是 `iostream` 中的 `basic_istream`  别名（`typedef`）。

```c++
MemoryInputStream::MemoryInputStream(const char* pBuffer, UInt32 bufferSize): 
    MemoryIOS(const_cast<char*>(pBuffer), bufferSize), istream(rdbuf()) {
}
 
MemoryInputStream::MemoryInputStream(MemoryInputStream& other):
    MemoryIOS(other), istream(rdbuf()) {
}
 
MemoryInputStream::~MemoryInputStream() {
}
```

唯一的一个成员函数是 `current`，封装了 `MemoryIOS` 的 `MemoryStreamBuf` 成员的 `gCurrent` 函数：

```c++
inline char* MemoryInputStream::current() {
    return rdbuf()->gCurrent();
}
```

#### 4、输出流

```c++
class MemoryOutputStream: public MemoryIOS, public std::ostream
    /// An input stream for reading from a memory area.
{
public:
    MemoryOutputStream(char* pBuffer,Poco::UInt32 bufferSize);
        /// Creates a MemoryOutputStream for the given memory area,
        /// ready for writing.
    MemoryOutputStream(MemoryOutputStream&);
    ~MemoryOutputStream();
        /// Destroys the MemoryInputStream.
 
    Poco::UInt32    written();
    void            written(Poco::UInt32 size);
    char*           current();
};
```

##### 4.1 构造函数、拷贝构造函数和析构函数

如下，不赘述了。

```c++
MemoryOutputStream::MemoryOutputStream(char* pBuffer, UInt32 bufferSize): 
    MemoryIOS(pBuffer, bufferSize), ostream(rdbuf()) {
}
MemoryOutputStream::MemoryOutputStream(MemoryOutputStream& other):
    MemoryIOS(other), ostream(rdbuf()) {
}
 
MemoryOutputStream::~MemoryOutputStream(){
}
```

##### 4.2 读取和设定已写字节数

读取：

```c++
inline Poco::UInt32 MemoryOutputStream::written() {
    return rdbuf()->written();
}
```

设定：

```c++
inline void MemoryOutputStream::written(Poco::UInt32 size) {
    rdbuf()->written(size);
}
```

##### 4.3 当前位置

与 `MemoryInputStream` 中的封装类似：

```c++
inline char* MemoryOutputStream::current() {
    return rdbuf()->pCurrent();
}
```

### 三、局部内存片

在第一部分的流缓冲区介绍 `MemoryStreamBuf` 时，其中有一个名为 `ScopedMemoryClip` 的友元，它就是本文所要介绍的。首先，最重要的是，`ScopedMemoryClip` 中有一个 `MemoryStreamBuf` 成员。

```c++
class ScopedMemoryClip {
public:
    ScopedMemoryClip(MemoryStreamBuf& buffer,Poco::UInt32 offset);
    ~ScopedMemoryClip();
private:
    void                clip(Poco::Int32 offset);
    Poco::UInt32        _offset;
    MemoryStreamBuf&   _buffer;
};
```

#### 1、构造函数

构造函数传入的参数对应的就是 `ScopedMemoryClip` 的两个成员值。其中偏移量不能超过 `MemoryStremamBuf` 的缓冲区上线值。

```c++
ScopedMemoryClip::ScopedMemoryClip(MemoryStreamBuf& buffer, UInt32 offset)
    : _offset(offset), _buffer(buffer) {
    if (_offset >= _buffer._bufferSize)
        _offset = _buffer._bufferSize - 1;
    if (_offset < 0)
        _offset = 0;
    clip(_offset);
}
```

#### 2、析构函数

```c++
ScopedMemoryClip::~ScopedMemoryClip() {
    clip(-(Int32)_offset);
}
```

#### 3、缓冲区切割

可以看到构造函数和析构函数中都调用了 `clip` 函数，该函数切割完缓冲区，形成局部内存片：

* 如果传入的偏移量参数为正，则仅保留切割之后的后一部分。
* 如果传入的参数为负，则相当于向前扩充缓冲区（只发生于析构函数中）。其源码如下。

```c++
void ScopedMemoryClip::clip(Int32 offset) {
 
    // 获取到 gptr
    char* gpos = _buffer.gCurrent();
 
    // 偏移缓冲区地址，并修改缓冲区大小
    _buffer._pBuffer += offset;
    _buffer._bufferSize -= offset;
 
    // pptr 的位置减去缓冲区新地址，作为 pptr 的新位置
    int ppos = _buffer.pCurrent() - _buffer._pBuffer;
 
    // 设置 gptr 可达区域和位置
    _buffer.setg(_buffer._pBuffer, gpos, _buffer._pBuffer + _buffer._bufferSize);
 
    // 设置 pptr 可达区域和位置
    _buffer.setp(_buffer._pBuffer, _buffer._pBuffer + _buffer._bufferSize);
    _buffer.pbump(ppos);
 
    // 如果已写数据数小于偏移量，则可以将已写数据数设置为零
    if (_buffer._written < offset)
        _buffer._written = 0;
 
    // 如果已写数据数大于等于偏移量，则减去 offset
    else
        _buffer._written -= offset;
 
    // 若已写字节数大于缓冲区容量，则设定为缓冲区容量
    if (_buffer._written > _buffer._bufferSize)
        _buffer._written = _buffer._bufferSize;
}
```

### Reference

1. http://www.cplusplus.com/reference/iostream/streambuf/gbump/
2. http://www.cplusplus.com/reference/iostream/streambuf/pbump/
3. http://www.cplusplus.com/reference/iostream/ios/init/