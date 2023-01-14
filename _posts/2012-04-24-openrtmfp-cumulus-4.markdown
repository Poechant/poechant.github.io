---
layout: post
title:  麦克船长的 OpenRTMFP/Cumulus 原理、源码及实践 4：AMF 解析源码分析
date:   2012-04-24 10:04:55 +0800
categories: rt_tech
tags: [直播技术]
description: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本篇文章主要介绍 ActionScript 独有的 AMF 数据格式，并对其序列化和反序列化的源码进行详细解读。
excerpt: 本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本篇文章主要介绍 ActionScript 独有的 AMF 数据格式，并对其序列化和反序列化的源码进行详细解读。
location: 广州
author: 麦克船长
---

**本文目录**
* TOC
{:toc}

本文是麦克船长《OpenRTMFP/Cumulus 原理、源码及实践》系列文章的其中一篇，相关内容最初首发于 CSDN 的 Poechant 技术博客，后整理于本博客。本篇文章主要介绍 ActionScript 独有的 AMF 数据格式，并对其序列化和反序列化的源码进行详细解读。

### 一、AMF 数据类型定义

#### 1、数据类型

各种数据类型的标示都在 AMF.h 中定义为宏

```c++
#define AMF_NUMBER              0x00    // 浮点数
#define AMF_BOOLEAN             0x01    // 布尔型
#define AMF_STRING              0x02    // 字符串
#define AMF_BEGIN_OBJECT        0x03    // 对象，开始
#define AMF_NULL                0x05    // null
#define AMF_UNDEFINED           0x06
#define AMF_REFERENCE           0x07
#define AMF_MIXED_ARRAY         0x08
#define AMF_END_OBJECT          0x09    // 对象，结束
#define AMF_BEGIN_TYPED_OBJECT  0x10
#define AMF_STRICT_ARRAY        0x0A
#define AMF_DATE                0x0B    // 日期
#define AMF_LONG_STRING         0x0C    // 字符串
#define AMF_UNSUPPORTED         0x0D
 
#define AMF_AVMPLUS_OBJECT  0x11
#define AMF_END             0xFF
 
#define AMF3_UNDEFINED      0x00
#define AMF3_NULL           0x01
#define AMF3_FALSE          0x02
#define AMF3_TRUE           0x03
#define AMF3_INTEGER        0x04
#define AMF3_NUMBER         0x05
#define AMF3_STRING         0x06
#define AMF3_DATE           0x08
#define AMF3_ARRAY          0x09
#define AMF3_OBJECT         0x0A
#define AMF3_BYTEARRAY      0x0C
#define AMF3_DICTIONARY     0x11
```

并定义了一个枚举类表示数据类型：

```c++
class AMF {
public:
    enum Type {
        Null=0,
        Boolean,
        Integer,
        Number,
        String,
        Date,
        Array,
        Object,
        ByteArray,
        Dictionary,
        RawObjectContent,
        End
    };
};
```

#### 2、`undefined` Type

`undefined` 类型由 `undefined` 类型标记表示。此值不会编码任何其他信息。

#### 3、`null` Type

`null` 类型由 `null` 类型标记表示。此值不会编码任何其他信息。

#### 4、`false` type

`false` 类型由 `false` 类型标记表示，用于编码布尔值 `false`。注意，在 ActionScript 3.0 中，布尔值的原始形式和对象形式不存在。此值不会编码任何其他信息。

#### 5、`true` type

true 类型由 true 类型标记表示，用于编码布尔值 true。注意，在 ActionScript 3.0 中，布尔值的原始形式和对象形式不存在。此值不会编码任何其他信息。

#### 6、`integer` type

在 AMF 3 中，整数使用可变长度的无符号 29 位整数进行序列化。ActionScript 3.0 中的整数类型 - 有符号 `int` 类型和无符号 `uint` 类型 - 也使用 29 位在 AVM+中表示。如果无符号整数 (`uint`) 的值大于等于 229 或者如果有符号整数 (`int`) 的值大于等于 228，则它将被 AVM+ 表示为 `double` 类型，并使用 AMF 3 double 类型进行序列化。

#### 7、`double` type

AMF 3 的 `double` 类型与 AMF 0 的 `Number` 类型编码方式相同。此类型用于编码 ActionScript `Number` 或值大于等于 228 的 ActionScript `int` 或值大于等于 229 的 ActionScript `uint`。编码值始终是网络字节顺序中的 8 字节 IEEE-754 双精度浮点值 (低内存中的符号位)。

#### 8、`String` type

ActionScript String 值使用 AMF 3 中的单个 string 类型表示 - AMF 0 中的 `string` 和 `long string` 类型的概念不再使用。可以使用对隐式字符串引用表中的索引将字符串作为先前发生的字符串的引用发送。字符串使用 UTF-8 编码 - 但是头可以描述字符串文本或字符串引用。空字符串永远不会作为引用发送。

#### 9、`XMLDocument` type

ActionScript 3.0 引入了新的 XML 类型 (参见 3.13)，但是旧版的 XMLDocument 类型在语言中被保留为 `flash.xml.XMLDocument`。与 AMF 0 类似，`XMLDocument` 的结构需要扁平化为字符串表示以进行序列化。与 AMF 中的其他字符串一样，内容使用 UTF-8 编码。XMLDocuments 可以通过使用对隐式对象引用表中的索引作为先前发生的 `XMLDocument` 实例的引用发送。

#### 10、`Date` type

在 AMF 3 中，ActionScript Date 简单地作为自 1970 年 1 月 1 日午夜 (UTC 时区) 以来的毫秒数进行序列化。不发送本地时区信息。可以使用对隐式对象引用表中的索引将日期作为先前发生的日期实例的引用发送。

#### 11、`Array` type

ActionScript 数组的类型和在数组中的位置是基于它们的索引性质描述的。以下表格概述了这些术语的含义：

* `strict`：仅包含序数（数字）索引
* `dense`：序数索引从 0 开始，并且在连续索引之间不存在间隙（即，从 0 到数组长度的每一个索引都被定义了）
* `sparse`：包含至少两个索引之间的一个间隙
* `associative`：包含至少一个非序数（字符串）索引（有时称为 ECMA 数组）

AMF 将数组分为两部分，密集部分和关联部分。关联部分的二进制表示由名称/值对（可能没有）终止的空字符串。密集部分的二进制表示由密集部分的大小（可能为零）以及有序的值列表（可能没有）组成。在 AMF 中写入的顺序是密集部分的大小，一个以空字符串终止的名称/值对列表，然后是大小的值。数组可以通过使用隐式对象引用表的索引作为先前发生的数组的引用来发送。

#### 12、`Object` type

AMF 3 中有一种类型用于处理 ActionScript 对象和自定义用户类。使用术语 "traits" 来描述类的定义特征。除了 "anonymous" 对象和 "typed" 对象，ActionScript 3.0 还引入了两个进一步的 traits 来描述如何序列化对象，即 "dynamic" 和 "externalizable"。以下表格概述了这些术语和它们的含义：

* `Anonymous`：实际的 ActionScript 对象类型的实例或没有注册别名的类的实例（在反序列化时将其视为对象）。
* `Typed`：具有注册别名的类的实例。
* `Dynamic`：具有动态特征声明的类定义的实例；可以在运行时动态地从实例中添加和删除公共变量成员。
* `Externalizable`：实现 flash.utils.IExternalizable 的类的实例，它完全控制其成员的序列化（特征信息中不包含属性名）。

在这些特征之外，对象的特征信息还可能包括在类上定义的一组公共变量和公共可读写属性名称（即不是函数的公共成员）。成员名称的顺序很重要，因为在特征信息之后的成员值将按照完全相同的顺序出现。这些成员被视为密封成员，因为它们是由类型明确定义的。

如果类型是动态的，则在密封成员之后可以包括一个进一步的部分，该部分将动态成员列为名称/值对。当遇到空字符串名称时，继续读取动态成员。

对象可以通过使用隐式对象引用表中的索引来作为先前发生对象的引用。此外，还可以通过使用隐式特征引用表中的索引将特征信息发送为先前发生的一组特征的引用。

#### 13、`XML` type

ActionScript 3.0 引入了一种新的 `XML` 类型，支持 E4X 语法。为了序列化，需要将 `XML` 类型展平成字符串表示形式。与 AMF 中的其他字符串一样，内容使用 UTF-8 编码。`XML` 实例可以通过使用对隐式对象引用表中的索引作为先前发生的 XML 实例的引用发送。请注意，这种编码对 `XML` 的使用造成了一些理论限制。每个 UTF-8 编码的 `XML` 实例的字节长度最大为 228-1 字节（大约 256 MB）。

#### 14、`ByteArray` type

用于保存字节数组，即 `ByteArray`。AMF 3 使用可变长度编码 29 位整数序列化此类型，其中包括字节长度前缀，然后是 `ByteArray` 的原始字节。`ByteArray` 实例可以通过使用对隐式对象引用表中的索引作为先前发生的 `ByteArray` 实例的引用发送。

#### 15、AMF3 的使用

##### 15.1、`NetConnection` and AMF 3

除了序列化 ActionScript 类型外，AMF 还可用于远程服务的异步调用。可使用简单的消息结构将一批请求发送到远程端点。此消息结构的格式为 AMF 0（参见[AMF0]）。可以使用特殊的 `avmplus-object-marker` 类型将上下文头值或消息正文切换到 AMF 3 编码。

##### 15.2、`NetConnection` in ActionScript 3.0

在 ActionScript 3.0 中，NetConnection 的限定类名是 flash.net.NetConnection。这个类仍然使用响应器来处理远程端点的结果和状态响应，但是现在需要强类型的 Responder 类。完全限定的类名是 flash.net.Responder。除了正常的结果和状态响应之外，NetConnection 还会分发事件，开发人员可以添加监听器。下面是这些事件的概述：

* 当异常异步抛出时触发，例如来自本机异步代码。
* 当输入或输出错误导致网络操作失败时触发。
* 当 NetConnection 对象报告其状态或错误条件时触发。
* 如果对 NetConnection.call() 的调用尝试连接到调用者安全沙箱外的服务器，则会触发。

##### 15.3、`ByteArray`, `IDataInput` and `IDataOutput`

ActionScript 3.0 引入了一种新类型，用于支持以字节数组形式处理原始数据，即 `flash.utils.ByteArray`。为了协助 ActionScript 对象序列化和复制，`ByteArray` 实现了 `flash.utils.IDataInput` 和 `flash.utils.IDataOutput`。这些接口指定了帮助将常见类型写入字节流的实用方法。两个感兴趣的方法是 `IDataOutput.writeObject` 和 `IDataInput.readObject`。这些方法使用 AMF 编码对象。使用的 AMF 版本由 `ByteArray.objectEncoding` 方法控制，该方法可以设置为 AMF 3 或 AMF 0。枚举类型 `flash.net.ObjectEncoding` 包含 AMF 版本的常量：分别为 `ObjectEncoding.AMF0` 和 `ObjectEncoding.AMF3`。

请注意，`ByteArray.writeObject` 使用一个版本的 AMF 对整个对象进行编码。与 `NetConnection` 不同，`ByteArray` 不会从 AMF 0 开始，然后将 `objectEncoding` 属性设置为 AMF 3 并切换到 AMF 3。还请注意，`ByteArray` 为每个 `readObject` 和 `writeObject` 调用使用新的对象、对象特征和字符串的隐式引用表。

### 二、`BinaryReader/Writer`

#### 1、AMF3 数据格式基础

首先介绍一下变长整数（Variable Length Integer），比如 UInt32 如下。

![image](/img/src/2012-04-24-openrtmfp-cumulus-4-1.png)

上图摘自 Adobe AMF3 官方文档，这是一种压缩方式的整数存储，且每一字节都对后面的数据具有预知作用。那么字符串如何处理呢？下面是字符串的处理方式，AMF0 和 AMF3 都才用 UTF-8 编码方式，并做如下压缩处理：

![image](/img/src/2012-04-24-openrtmfp-cumulus-4-2.png)

上图摘自 Adobe AMF3 官方文档。

#### 2、序列化

序列化包括 8 位、16 位、32 位，以及 UTF-8 和 UTF-16（I guess）编码的 String，还有原生数据（Raw Data）、变长无符号整数（Variable Length Unsigned Integer）以及 IP 地址。所谓序列化就是按照指定格式编写各种对象、基础数据类型值。

```c++
class BinaryWriter : public Poco::BinaryWriter {
public:
    BinaryWriter(std::ostream& ostr);
    virtual ~BinaryWriter();
    void writeRaw(const Poco::UInt8* value,Poco::UInt32 size);
    void writeRaw(const char* value,Poco::UInt32 size);
    void writeRaw(const std::string& value);
    void write8(Poco::UInt8 value);
    void write16(Poco::UInt16 value);
    void write32(Poco::UInt32 value);
    void writeString8(const std::string& value);
    void writeString8(const char* value,Poco::UInt8 size);
    void writeString16(const std::string& value);
    void writeString16(const char* value,Poco::UInt16 size);
    void write7BitValue(Poco::UInt32 value);
    void write7BitLongValue(Poco::UInt64 value);
    void writeAddress(const Address& address,bool publicFlag);
    void writeAddress(const Poco::Net::SocketAddress& address,bool publicFlag);
    static BinaryWriter BinaryWriterNull;
};
```

请注意其中名为 `BinaryWriterNull` 的成员。构造函数定义为：

```c++
BinaryWriter::BinaryWriter(ostream& ostr):
    Poco::BinaryWriter(ostr,BinaryWriter::NETWORK_BYTE_ORDER) {
}

BinaryWriter::~BinaryWriter() {
    flush();
}
```

其中 `writeRaw` 是简单地封装 `Poco::BinaryWriter::writeRaw()`，如下：

```c++
inline void BinaryWriter::writeRaw(const Poco::UInt8* value,Poco::UInt32 size) {
    Poco::BinaryWriter::writeRaw((char*)value,size);
}
inline void BinaryWriter::writeRaw(const char* value,Poco::UInt32 size) {
    Poco::BinaryWriter::writeRaw(value,size);
}
inline void BinaryWriter::writeRaw(const std::string& value) {
    Poco::BinaryWriter::writeRaw(value);
}
```

写入整数实现如下，用的是从 `Poco::BinaryReader` 继承来的重载运算符操作：

```c++
inline void BinaryWriter::write8(Poco::UInt8 value) {
    (*this) << value;
}   
inline void BinaryWriter::write16(Poco::UInt16 value) {
    (*this) << value;
}
inline void BinaryWriter::write32(Poco::UInt32 value) {
    (*this) << value;
}
```

写入字符串：

```c++
void BinaryWriter::writeString8(const char* value,UInt8 size) {
    write8(size);
    writeRaw(value,size);
}
void BinaryWriter::writeString8(const string& value) {
    write8(value.size());
    writeRaw(value);
}
void BinaryWriter::writeString16(const char* value,UInt16 size) {
    write16(size);
    writeRaw(value,size);
}
void BinaryWriter::writeString16(const string& value) {
    write16(value.size());
    writeRaw(value);
}
```

写入变长整数，这段代码含义也一目了然，就是读取变长无符号 32 位整数、64 位整数。

```c++
void BinaryWriter::write7BitValue(UInt32 value) {
    UInt8 shift = (Util::Get7BitValueSize(value)-1)*7;
    bool max = false;
    if(shift>=21) { // 4 bytes maximum
        shift = 22;
        max = true;
    }
 
    while(shift>=7) {
        write8(0x80 | ((value>>shift)&0x7F));
        shift -= 7;
    }
    write8(max ? value&0xFF : value&0x7F);
}
```

```c++
void BinaryWriter::write7BitLongValue(UInt64 value) {
    UInt8 shift = (Util::Get7BitValueSize(value)-1)*7;
    bool max = shift>=63; // Can give 10 bytes!
    if(max)
        ++shift;
 
    while(shift>=7) {
        write8(0x80 | ((value>>shift)&0x7F));
        shift -= 7;
    }
    write8(max ? value&0xFF : value&0x7F);
}
```

写入 IP 地址的两个函数暂略。

#### 3、反序列化

反序列化就是从指定格式的数据中读出各类型的数据值。

```c++
class BinaryReader : public Poco::BinaryReader {
public:
    BinaryReader(std::istream& istr);
    virtual ~BinaryReader();
 
    Poco::UInt32    read7BitValue();
    Poco::UInt64    read7BitLongValue();
    Poco::UInt32    read7BitEncoded();
    void            readString(std::string& value);
    void            readRaw(Poco::UInt8* value,Poco::UInt32 size);
    void            readRaw(char* value,Poco::UInt32 size);
    void            readRaw(Poco::UInt32 size,std::string& value);
    void            readString8(std::string& value);
    void            readString16(std::string& value);
    Poco::UInt8     read8();
    Poco::UInt16    read16();
    Poco::UInt32    read32();
    bool            readAddress(Address& address);
 
    static BinaryReader BinaryReaderNull;
};
```

构造与析构函数都很简单：

```c++
BinaryReader::BinaryReader(istream& istr) : Poco::BinaryReader(istr,BinaryReader::NETWORK_BYTE_ORDER) {
}
 
BinaryReader::~BinaryReader() {
}
```

读取原生数据（Raw Data）：

```c++
inline void BinaryReader::readRaw(Poco::UInt8* value,Poco::UInt32 size) {
    Poco::BinaryReader::readRaw((char*)value,size);
}
inline void BinaryReader::readRaw(char* value,Poco::UInt32 size) {
    Poco::BinaryReader::readRaw(value,size);
}
inline void BinaryReader::readRaw(Poco::UInt32 size,std::string& value) {
    Poco::BinaryReader::readRaw(size,value);
}
```

写整数，用的是 `Poco::BinaryWriter` 的重载运算符：

```c++
inline void BinaryWriter::write8(Poco::UInt8 value) {
    (*this) << value;
}
 
inline void BinaryWriter::write16(Poco::UInt16 value) {
    (*this) << value;
}
 
inline void BinaryWriter::write32(Poco::UInt32 value) {
    (*this) << value;
}
```

读写整数依旧使用从 `Poco::BinaryReader` 继承来的运算符操作：

```c++
UInt8 BinaryReader::read8() {
    UInt8 c;
    (*this) >> c;
    return c;
}
 
UInt16 BinaryReader::read16() {
    UInt16 c;
    (*this) >> c;
    return c;
}
 
UInt32 BinaryReader::read32() {
    UInt32 c;
    (*this) >> c;
    return c;
}
```

写字符串：

```c++
void BinaryWriter::writeString8(const char* value,UInt8 size) {
    write8(size);
    writeRaw(value,size);
}
void BinaryWriter::writeString8(const string& value) {
    write8(value.size());
    writeRaw(value);
}
void BinaryWriter::writeString16(const char* value,UInt16 size) {
    write16(size);
    writeRaw(value,size);
}
void BinaryWriter::writeString16(const string& value) {
    write16(value.size());
    writeRaw(value);
}
```

读取变长整数，分别针对 `UInt32` 和 `UInt64`，要理解 `AMF3` 的变长整数才能理解这个：

```c++
UInt32 BinaryReader::read7BitValue() {
    UInt8 n = 0;
    UInt8 b = read8();
    UInt32 result = 0;
    while ((b&0x80) && n < 3) {
        result <<= 7;
        result |= (b&0x7F);
        b = read8();
        ++n;
    }
    result <<= ((n<3) ? 7 : 8); // Use all 8 bits from the 4th byte
    result |= b;
    return result;
}
```

```c++
UInt64 BinaryReader::read7BitLongValue() {
    UInt8 n = 0;
    UInt8 b = read8();
    UInt64 result = 0;
    while ((b&0x80) && n < 8) {
        result <<= 7;
        result |= (b&0x7F);
        b = read8();
        ++n;
    }
    result <<= ((n<8) ? 7 : 8); // Use all 8 bits from the 4th byte
    result |= b;
    return result;
}
```

### 三、`PacketReader/Writer`

#### 1、PacketReader

```
#define PACKETRECV_SIZE     2048
class PacketReader: public BinaryReader {
public:
    PacketReader(const Poco::UInt8* buffer,Poco::UInt32 size);
    PacketReader(PacketReader&);
    virtual ~PacketReader();
    const Poco::UInt32  fragments;
    Poco::UInt32    available(); // 可读字节数
    Poco::UInt8*    current();
    Poco::UInt32    position(); // 获取当前的相对位置（相对于起始位置的）
    void            reset(Poco::UInt32 newPos = 0); // 设定当前位置
    void            shrink(Poco::UInt32 rest);
    void            next(Poco::UInt32 size);
private:
    MemoryInputStream _memory;
};
```

###### 1.1、封装 `MemoryInputStream`

`available`

```c++
  inline Poco::UInt32 PacketReader::available() {
      return _memory.available();
  }
```

`current`：当前绝对位置（内存地址）

```c++
  inline Poco::UInt8* PacketReader::current() {
      return (Poco::UInt8*)_memory.current();
  }
```

`position`：当前位置（绝对位置）减去缓冲区起始位置

```c++
  inline Poco::UInt32 PacketReader::position() {
      return _memory.current() - _memory.begin();
  }
```

`reset`

```c++
  inline void PacketReader::reset(Poco::UInt32 newPos) {
      _memory.reset(newPos);
  }
```

`next`

```c++
  inline void PacketReader::next(Poco::UInt32 size) {
      return _memory.next(size);
  }
```

###### 1.2、收缩缓冲区

封装了 `MemoryInputStream` 的 `resize`。不过由于前面的 `if` 语句影响，传给 resize 的参数一定不会大于缓冲区的当前大小。

```c++
void PacketReader::shrink(UInt32 rest) {
    if (rest > available()) {
        WARN("rest %u more upper than available %u bytes",rest,available());
        rest = available();
    }
    _memory.resize(position() + rest);
}
```

###### 1.3、构造函数、拷贝构造函数和析构函数

构造函数先调用父类 `BinaryReader` 的构造函数，并初始化 `fragments` 和 `_memory` 输入流的缓冲区。

```c++
PacketReader::PacketReader(const UInt8* buffer,UInt32 size)
    : _memory((const char*)buffer, size),
      BinaryReader(_memory),
      fragments(1) {
}
 
// Consctruction by copy
PacketReader::PacketReader(PacketReader& other)
    : _memory(other._memory),
      BinaryReader(_memory),
      fragments(other.fragments) {
}
 
PacketReader::~PacketReader() {
}
```

#### 2、`PacketWriter`

```c++
class PacketWriter: public BinaryWriter {
public:
    PacketWriter(const Poco::UInt8* buffer,Poco::UInt32 size);
    PacketWriter(PacketWriter&);
    virtual ~PacketWriter();
    Poco::UInt8*        begin();
    Poco::UInt32        length();
    Poco::UInt32        position();
    Poco::UInt32        available();
    bool    good();
    void    clear(Poco::UInt32 pos=0);
    void    reset(Poco::UInt32 newPos);
    void    limit(Poco::UInt32 length=0);
    void    next(Poco::UInt32 size);
    void    flush();
private:
    MemoryOutputStream  _memory;
    PacketWriter*       _pOther;
    Poco::UInt32        _size;
};
```

###### 2.1、封装`MemoryOutputStream`

`available`

```c++
  inline Poco::UInt32 PacketWriter::available() {
      return _memory.available();
  }
```

`good`：不过 `MemoryOutputStream` 也是封装的 `std::ostream` 的 `good` 函数，True if no error flags are set.

```c++
  inline bool PacketWriter::good() {
      return _memory.good();
  }
```

`written`

```c++
  inline Poco::UInt32 PacketWriter::length() {
      return _memory.written();
  }
```

`position`

```c++
  inline Poco::UInt32 PacketWriter::position() {
      return _memory.current()-(char*)begin();
  }
```

`reset`：设置缓冲区的指针位置，即 `position`

```c++
  inline void PacketWriter::reset(Poco::UInt32 newPos) {
      _memory.reset(newPos);
  }
```

`next`：移动缓冲区指针

```c++
  inline void PacketWriter::next(Poco::UInt32 size) {
      return _memory.next(size);
  }
```

`begin`：返回缓冲区的起始地址

```c++
  inline Poco::UInt8* PacketWriter::begin() {
      return (Poco::UInt8*)_memory.begin();
  }
```

`clear`：其实就是修改 written 和 position，使得指定位置后面的数据在以后写的时候可以被覆盖，并不是真正的清除。

```c++
  void PacketWriter::clear(UInt32 pos) {
      reset(pos);
      _memory.written(pos);
  }
```

`limit`：封装 `MemoryOutputStream` 的 `resize`

```c++
  void PacketWriter::limit(UInt32 length) {
      if (length == 0)
          length = _size;
      if (length > _size) {
          WARN("Limit '%d' more upper than buffer size '%d' bytes",length,_size);
          length = _size;
      }
      _memory.resize(length);
  }
```

###### 2.2、封装 `BinaryWriter`

`flush`：封装 `BinaryWriter` 的 `flush`，不过 `BinaryWriter` 的 `flush` 实际上是从 `Poco::BinaryWriter` 继承而来的。其作用是「Flushes the underlying stream」。

```c++
  void PacketWriter::flush() {
      if (_pOther && _memory.written() > _pOther->_memory.written())
          _pOther->_memory.written(_memory.written());
      BinaryWriter::flush();
  }
```

###### 2.3、构造函数、拷贝构造函数和析构函数

```c++
PacketWriter::PacketWriter(const UInt8* buffer, UInt32 size)
    : _memory((char*)buffer, size),
      BinaryWriter(_memory),
      _pOther(NULL),
      _size(size) {
}
 
// Consctruction by copy
PacketWriter::PacketWriter(PacketWriter& other)
    : _pOther(&other),
      _memory(other._memory),
      BinaryWriter(_memory),
      _size(other._size) {
}
```

注意析构函数中会进行 `flush`：

```c++
PacketWriter::~PacketWriter() {
    flush();
}
```

### 四、`AMFReader`

#### 1、`ObjectDef`

```c++
class ObjectDef {
public: 
    ObjectDef(UInt32 amf3,UInt8 arrayType=0)
        : amf3(amf3),
          reset(0),
          dynamic(false),
          externalizable(false),
          count(0),
          arrayType(arrayType) {
    }
 
    list<string>    hardProperties;
    UInt32          reset;
    bool            dynamic;
    bool            externalizable;
    UInt32          count;
    UInt8           arrayType;
    const UInt32    amf3;
};
```

#### 2、`AMFReader` 定义

其中 `PacketReader` 作为其成员。

```c++
class AMFReader {
public:
    AMFReader(PacketReader& reader);
    ~AMFReader();
 
    void            readSimpleObject(AMFSimpleObject& object);
 
    void            read(std::string& value);
    double          readNumber();
    Poco::Int32     readInteger();
    bool            readBoolean();
    BinaryReader&   readByteArray(Poco::UInt32& size);
    Poco::Timestamp readDate();
 
    bool            readObject(std::string& type);
    bool            readArray();
    bool            readDictionary(bool& weakKeys);
    AMF::Type       readKey();
    AMF::Type       readValue();
    AMF::Type       readItem(std::string& name);
    BinaryReader&   readRawObjectContent();
 
    void            readNull();
    AMF::Type       followingType();
 
    bool            available();
 
    void            startReferencing();
    void            stopReferencing();
 
    PacketReader&   reader;
 
private:
    void                            readString(std::string& value);
    Poco::UInt8                     current();
    void                            reset();
    std::list<ObjectDef*>           _objectDefs;
    std::vector<Poco::UInt32>       _stringReferences;
    std::vector<Poco::UInt32>       _classDefReferences;
    std::vector<Poco::UInt32>       _references;
    std::vector<Poco::UInt32>       _amf0References;
    Poco::UInt32                    _amf0Reset;
    Poco::UInt32                    _reset;
    Poco::UInt32                    _amf3;
    bool                            _referencing;
};
```

##### 2.1、构造函数、析构函数

参数为 `PacketReader`，会初始化一些成员变量。

```c++
AMFReader::AMFReader(PacketReader& reader)
    : reader(reader),
      _reset(0),
      _amf3(0),
      _amf0Reset(0),
      _referencing(true) {
}
```

析构时，会逐一释放 `_objectDefs` 中对象的内存：

```c++
AMFReader::~AMFReader() {
    list<ObjectDef*>::iterator it;
    for (it = _objectDefs.begin(); it!=_objectDefs.end(); ++it)
        delete *it;
}
```

##### 2.2、简单封装 `PacketReader` 的一些函数

`reset`：操作指针位置

```c++
  void AMFReader::reset() {
      if (_reset > 0) {
          reader.reset(_reset);
          _reset = 0;
      }
  }
```

`available`：根据当前缓冲区大小和 `written` 计算得到

```c++
  bool AMFReader::available() {
      reset();
      return reader.available() > 0;
  }
```

`current`：`gptr` 内存地址

```c++
  inline Poco::UInt8 AMFReader::current() {
      return *reader.current();
  }
```

##### 2.3、设置 `gptr` 位置

其实 `pptr` 也被影响了，但是在 `AMFReader` 中只用 `gptr`。调用构造函数的时候，`reset` 被设为 0，其后在每次读取数据的时候都会影响 `reset`。

```c++
void AMFReader::reset() {
    if(_reset>0) {
        reader.reset(_reset);
        _reset=0;
    }
}
```

##### 2.4、判断类型

分析请看注释：

```c++
AMF::Type AMFReader::followingType() {
```

先 `reset`：

```c++
    reset();
    if (_amf3 != reader.position()) {
        if (_objectDefs.size() > 0)
            _amf3 = _objectDefs.back()->amf3;
```

是 AMF0 类型：

```c++
        else
            _amf3 = 0;
    }
```

如果没有可读数据了，则返回 AMF::End。

```c++
    if (!available())
        return AMF::End;
```

开始读了，先读到的表示 AMF 数据类型。要注意的是调用 current 并不改变指针的位置，所以你会在线面看到调用 next。

```c++
    UInt8 type = current();
 
    if (!_amf3 && type == AMF_AVMPLUS_OBJECT) {
        reader.next(1);
        _amf3 = reader.position();
        if(!available())
            return AMF::End;
        type = current();
    }
```

AMF3 类型

```c++
    if (_amf3) {
        switch(type) {
```

Undefined 和 null 都当做 null。

```c++
            case AMF3_UNDEFINED:
            case AMF3_NULL:
                return AMF::Null;
```

false 和 true 都是 boolean。

```c++
            case AMF3_FALSE:
            case AMF3_TRUE:
                return AMF::Boolean;
            case AMF3_INTEGER:
                return AMF::Integer;
            case AMF3_NUMBER:
                return AMF::Number;
            case AMF3_STRING:
                return AMF::String;
            case AMF3_DATE:
                return AMF::Date;
            case AMF3_ARRAY:
                return AMF::Array;
            case AMF3_DICTIONARY:
                return AMF::Dictionary;
            case AMF3_OBJECT:
                return AMF::Object;
            case AMF3_BYTEARRAY:
                return AMF::ByteArray;
```

落到 default 手里的话，就跳过这个字节，读取下一个。

```c++
            default:
                ERROR("Unknown AMF3 type %.2x",type)
                reader.next(1);
                return followingType();
        }
    }
```

AMF0 类型

```c++
    switch (type) {
```

`undefined` 和 `null` 都是 `null`

```c++
        case AMF_UNDEFINED:
        case AMF_NULL:
            return AMF::Null;
 
        case AMF_BOOLEAN:
            return AMF::Boolean;
        case AMF_NUMBER:
            return AMF::Number;
```

`long string` 和 `string` 都是 `string`

```c++
        case AMF_LONG_STRING:
        case AMF_STRING:
            return AMF::String;
```

`mixed array` 和 `strict array` 都是 `array`

```c++
        case AMF_MIXED_ARRAY:
        case AMF_STRICT_ARRAY:
            return AMF::Array;
 
        case AMF_DATE:
            return AMF::Date;
```

`begin object` 和 `begin typed object` 都是 `object`

```c++
        case AMF_BEGIN_OBJECT:
        case AMF_BEGIN_TYPED_OBJECT:
            return AMF::Object;
```

如果是引用，就跳过表示类型值的这个字节。这个先留下来，带我们分析完 readArray 和 readObject 再回头看。

```c++
        case AMF_REFERENCE: {
            reader.next(1);
            UInt16 reference = reader.read16();
            if (reference > _amf0References.size()) {
                ERROR("AMF0 reference not found")
                return followingType();
            }
            _amf0Reset = reader.position();
            reader.reset(_amf0References[reference]);
            return followingType();
        }
```

如果没了，或者不支持，或者都不是，就跳过这个字节，递归继续读取：

```c++
        case AMF_END_OBJECT:
            ERROR("AMF end object type without begin object type before")
            reader.next(1);
            return followingType();
        case AMF_UNSUPPORTED:
            WARN("Unsupported type in AMF format")
            reader.next(1);
            return followingType();
        default:
            ERROR("Unknown AMF type %.2x",type)
            reader.next(1);
            return followingType();
    }
}
```

`followingType` 是这个类的核心，每个具体的数据类型的分析都依赖于它的判断。这些类型的解析，会在下一篇文章中介绍。

#### 3、解析 AS3 `Null`

```c++
void AMFReader::readNull() {
```

先 reset 一下是惯例，就像糗百上的割一样。。

```c++
    reset(); 
```

AMF 数据类型

```c++
    AMF::Type type = followingType();
```

如果是 `Null`，跳过该字节，并返回

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return;
    }
```

判断错误

```c++
    ERROR("Type %.2x is not a AMF Null type",type);
}
```

#### 4、解析 AS3 `Number`

```c++
double AMFReader::readNumber() {
```

惯例：

```c++
    reset();
```

类型：

```c++
    AMF::Type type = followingType();
```

`Null` 会被悲催的跳过：

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return 0;
    }
```

不是 `Number` 呀 :(

```c++
    if (type != AMF::Number) {
        ERROR("Type %.2x is not a AMF Number type",type);
            return 0;
    }
```

跳过该字节吧

```c++
    reader.next(1);
```

返回吧，返回之前还用到 `Poco::BinaryReader` 的运算符，注意 AS3 中的 `Number` 就是 C++ 的 `double`。

```c++
    double result;
    reader >> result;
    return result;
}
```

#### 5、解析 AS3 `Integer`

```c++
Int32 AMFReader::readInteger() {
```

`reset` 类型：

```c++
    reset();
    AMF::Type type = followingType();
```

Null 的话：

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return 0;
    }
```

不是 `Integer` 或者 Number 的话。。。

```c++
    if (type != AMF::Integer && type != AMF::Number) {
        ERROR("Type %.2x is not a AMF Integer type",type);
        return 0;
    }
```

跳过吧。

```c++
    reader.next(1);
```

终于是 `Number` 了。

```c++
    if (type == AMF::Number) {
        double result;
        reader >> result;
        return (Int32)result;
    }
```

读一个变长的 32 位无符号整数：

```c++
    // Forced in AMF3 here!
    UInt32 value = reader.read7BitValue();
```

如果大于 3.5 个字节所能表示的最大无符号整数值（`268435455` 是 `0xFFFFFFF`），则减去 `0x2FFFFFFF`（这还不是太理解，有能解释的朋友给留个言，或者发 email 给我 ）

```c++
    if (value > 268435455)
        value -= (1 << 29);
    return value;
}
```

#### 6、解析 AS3 `Boolean`

```c++
bool AMFReader::readBoolean() {
```

惯例：

```c++
    reset();
    AMF::Type type = followingType();
```

如果是 `Null`：

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return false;
    }
```

居然不是 `Boolean` 的话。。

```c++
    if (type != AMF::Boolean) {
        ERROR("Type %.2x is not a AMF Boolean type",type);
        return false;
    }
```

如果是 `AMF3` 的话，返回 `true` 或者 `false`：

```c++
    if (_amf3)
        return reader.read8()== AMF3_FALSE ? false : true;
```

不是 `AMF3` 就是 `AMF0` 喽：

```c++
    reader.next(1);
    return reader.read8()==0x00 ? false : true;
}
```

#### 7、开始引用与结束引用

如下这两个函数会在 `FlowConnection` 中调用。

```c++
inline void AMFReader::startReferencing() {
    _referencing = true;
}
 
inline void AMFReader::stopReferencing() {
    _referencing = false;
}
```

#### 8、解析 AS3 `ByteArray`

先回顾一下 AMF3 中的ByteArray 的数据格式：

注意到，首先要读取一个变长无符号 32 位整数，但是最低位是 1，只有 28 位用于表示数据长度。解释完这里，下面的解析过程才好理解。

```c++
BinaryReader& AMFReader::readByteArray(UInt32& size) {
```

惯例：

```c++
    reset();
    AMF::Type type = followingType();
```

`Null` 就返回 `BinaryReaderNull`。

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return BinaryReader::BinaryReaderNull;
    }
```

如果不是 `ByteArray`，也返回 `BinaryReaderNull`：

```c++
    if (type != AMF::ByteArray) {
        ERROR("Type %.2x is not a AMF ByteArray type",type);
        return BinaryReader::BinaryReaderNull;
    }
```

跳过这个字节：

```c++
    reader.next(1);
```

注意 position 返回的是相对位置，与 AS3 中一样。`reference` 表示这个地址（简单说，引用就是地址嘛）。

```c++
    UInt32 reference = reader.position();
```

读取一个变长 32 位无符号整数：

```c++
    size = reader.read7BitValue();
```

最低位是 1 的话，`isInline` 是 `true`，否则为 `false`。

```c++
    bool isInline = size & 0x01;
```

右移一位，因为那一位是标志位，上面解释过了。

```c++
    size >>= 1;
```

如果 `isInline` 是 `true`，表示是 `ByteArray`：

```c++
    if (isInline) {
```

如果 `_referencing` 为 `true` 的话（这是一个 `vector`），push back this reference：

```c++
        if (_referencing)
            _references.push_back(reference);
    }
```

不符合 ByteArray 的格式定义的话：

```c++
    else {
        if (size > _references.size()) {
            ERROR("AMF3 reference not found")
            return BinaryReader::BinaryReaderNull;
        }
        _reset = reader.position();
```

移动到这个 reference 的位置，`_references[size]` 就是这个位置（相对）。

```c++
        reader.reset(_references[size]); // TODO size 作为索引，还没有完全理解
```

读取这个 reference 的 size 值给 size对象（注意 size 是这个函数传入的引用参数，其值可以被修改）。

```c++
        size = reader.read7BitValue() >> 1;
    }
```

把读取完 ByteArraty 的 PacketReader 返回：

```c++
    return reader;
}
```

最后强调一点，`ByteArray` 的数据段最大长度为 228 -1 字节，约为 256 MB。

#### 9、解析 AS3 `Date`

先看下 `Date` 的数据格式：

下面开始分析：

```c++
Timestamp AMFReader::readDate() {
```

惯例：

```c++
    reset();
    AMF::Type type = followingType();
```

Null 的话，就返回当前时间：

```c++
    if (type == AMF::Null) {
        reader.next(1);
        return Timestamp(0);
    }
```

如果不是 Date 类型，也返回当前时间：

```c++
    if (type != AMF::Date) {
        ERROR("Type %.2x is not a AMF Date type",type);
        return Timestamp(0);
    }
 
    reader.next(1);
    double result = 0;
```

如果是 AMF3：

```c++
    if (_amf3) {
```

先读取 flag，最低一位必须是 1，其他位丢到垃圾桶。

```c++
        UInt32 flags = reader.read7BitValue();
```

当前相对位置。

```c++
        UInt32 reference = reader.position();
```

是 1 就 push back 到 `_references` 里。

```c++
        bool isInline = flags & 0x01;
        if (isInline) {
            if(_referencing)
                _references.push_back(reference);
```

读取一个 double，到 result 里（result 也是 double 类型哦~）。

```c++
            reader >> result;
        }
```

如果标志位不是 1，麻烦不少哒。。。

```c++
        else {
            flags >>= 1;
```

如果 flag 超了，就返回当前时间作为时间戳作为 Date。

```c++
            if (flags > _references.size()) {
                ERROR("AMF3 reference not found")
                return Timestamp(0);
            }
```

这段与 ByteArray 那段一样：

```c++
            _reset = reader.position();
            reader.reset(_references[flags]);
            reader >> result;
            reset();
        }
```

返回喽~

```c++
        return Timestamp((Timestamp::TimeVal) result * 1000);
    }
    reader >> result;
```

读俩，因为是 double（64 位）：

```c++
    reader.next(2); // Timezone, useless
```

返回喽~

```c++
    return Timestamp((Timestamp::TimeVal) result * 1000);
}
```

#### 10、解析 AS3 `Dictionary`

```c++
bool AMFReader::readDictionary(bool& weakKeys) {
```

下面这段咱就略了。。

```c++
    reset();
    AMF::Type type = followingType();
    if (type == AMF::Null) {
        reader.next(1);
        return false;
    }
    if (type != AMF::Dictionary) {
        ERROR("Type %.2x is not a AMF Dictionary type",type);
        return false;
    }
```

跳过 type：

```c++
    // AMF3
    reader.next(1); // marker
```

当前相对位置值作为 `reference`，再读个 `size`，还是最低位必须为 1，不是就返回 `false`。

```c++
    UInt32 reference = reader.position();
    UInt32 size = reader.read7BitValue();
    bool isInline = size & 0x01;
    size >>= 1;
    if(!isInline && size>_references.size()) {
        ERROR("AMF3 reference not found")
        return false;
    }
```

下面要调用到 `ObjectRef` 构造函数，这里再把其实现拿出来看看，其实主要是初始化了哪些成员。

```c++
ObjectDef(UInt32 amf3,UInt8 arrayType=0)
    : amf3(amf3),
      reset(0),
      dynamic(false),
      externalizable(false),
      count(0),
      arrayType(arrayType) {
}
```

可以看到要有一个 amf3，还有 `reset` 置为 0，`dynamic` 置为 `false`，`externalizable` 也是 `false`，`count` 是 0，`arrayType` 成员要赋值。

上面是插播哦，下面还要继续哒。创建这么一个对象，注意是 new 出来的，所以我们在《OpenRTMFP/Cumulus Primer（16）AMF解析之AMFReader》一文中提到了 AMFReader 的析构函数中要对 `_objectRef` 的每个元素逐一析构的。`arrayType` 就设置为 `AMF3_DICTIONARY`。

```c++
    ObjectDef* pObjectDef = new ObjectDef(_amf3, AMF3_DICTIONARY);
    pObjectDef->dynamic=true;
    _objectDefs.push_back(pObjectDef);
```

如果标志位是 1，就直接 push back，跟之前一样。不过这里多了一个 `pObjectDef`，所以还要设置一下它的计数为 `size`，就是 `dictionary` 数据大小。

```c++
    if (isInline) {
        if (_referencing)
            _references.push_back(reference);
        pObjectDef->count = size;
    }
```

如果标志位是 0，就把 `count` 设置为下一个变长整数值。

```c++
    else {
        pObjectDef->reset = reader.position();
        reader.reset(_references[size]);
        pObjectDef->count = reader.read7BitValue() >> 1;
    }
    pObjectDef->count *= 2;
```

读一个字节，如果最小位是 1，weakKeys 就是 true，否则为 false。

```c++
    weakKeys = reader.read8() & 0x01;
 
    return true;
}
```