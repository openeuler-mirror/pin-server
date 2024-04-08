/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may
   not use this file except in compliance with the License. You may obtain
   a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
   License for the specific language governing permissions and limitations
   under the License.

   Author: Guangya Ding
   Create: 2022-11-23
   Description:
    This file defines the types for the Plugin dialect in MLIR. These MLIR types
    correspond to the Plugin IR type system.
*/

#ifndef MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_
#define MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace PluginIR {
using namespace mlir;

enum PluginTypeID {
    UndefTyID = 0,
    // PrimitiveTypes
    VoidTyID,      ///< type with no size
    UIntegerTy1ID,  ///< 1-bit unsigned integer type
    UIntegerTy8ID,  ///< 8-bit unsigned integer type
    UIntegerTy16ID,  ///< 16-bit unsigned integer type
    UIntegerTy32ID,  ///< 32-bit unsigned integer type
    UIntegerTy64ID,  ///< 64-bit unsigned integer type
    IntegerTy1ID,  ///< 1-bit signed integer type
    IntegerTy8ID,  ///< 8-bit signed integer type
    IntegerTy16ID,  ///< 16-bit signed integer type
    IntegerTy32ID,  ///< 32-bit signed integer type
    IntegerTy64ID,  ///< 64-bit signed integer type
    BooleanTyID,    ///< 1-bit signless integer type
    FloatTyID,     ///< 32-bit floating point type
    DoubleTyID,    ///< 64-bit floating point type

    // Derived types
    FunctionTyID,      ///< Functions
    PointerTyID,       ///< Pointers
    StructTyID,        ///< Structures
    ArrayTyID,         ///< Arrays
    VectorTyID,         ///< Arrays
};

class PluginTypeBase : public Type {
public:
    using Type::Type;
    /// FIXME.
    static bool classof(Type type) { return true; }

    PluginTypeID getPluginTypeID ();
    unsigned getPluginIntOrFloatBitWidth ();
    bool isSignedPluginInteger ();
    bool isUnsignedPluginInteger ();
    void setTypeSize (unsigned size);
    unsigned getTypeSize ();

private:
    unsigned size;
}; // class PluginTypeBase

namespace detail {
    struct PluginIntegerTypeStorage;
    struct PluginFloatTypeStorage;
    struct PluginPointerTypeStorage;
    struct PluginTypeAndSizeStorage;
    struct PluginFunctionTypeStorage;
    struct PluginStructTypeStorage;
}

class PluginIntegerType : public Type::TypeBase<PluginIntegerType, PluginTypeBase, detail::PluginIntegerTypeStorage> {
public:
    using Base::Base;

    enum SignednessSemantics {
        Signless, /// No signedness semantics
        Signed,   /// Signed integer
        Unsigned, /// Unsigned integer
    };
    /// Return true if this is a signless integer type.
    bool isSignless() const { return getSignedness() == Signless; }
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getSignedness() == Signed; }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return getSignedness() == Unsigned; }

    PluginTypeID getPluginTypeID ();

    static PluginIntegerType get(MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);

    unsigned getWidth() const;
    SignednessSemantics getSignedness() const;
};

class PluginFloatType : public Type::TypeBase<PluginFloatType, PluginTypeBase, detail::PluginFloatTypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static PluginFloatType get(MLIRContext *context, unsigned width);

    unsigned getWidth() const;
};

class PluginPointerType : public Type::TypeBase<PluginPointerType, PluginTypeBase, detail::PluginPointerTypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static PluginPointerType get(MLIRContext *context, Type pointee, unsigned readOnlyPointee = 0);

    Type getElementType();

    unsigned isReadOnlyElem();
}; // class PluginPointerType

class PluginArrayType : public Type::TypeBase<PluginArrayType, PluginTypeBase, detail::PluginTypeAndSizeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static bool isValidElementType(Type type);

    static PluginArrayType get(MLIRContext *context, Type elementType, unsigned numElements);

    Type getElementType();

    unsigned getNumElements();
}; // class PluginArrayType

class PluginVectorType : public Type::TypeBase<PluginVectorType, PluginTypeBase, detail::PluginTypeAndSizeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static bool isValidElementType(Type type);

    static PluginVectorType get(MLIRContext *context, Type elementType, unsigned numElements);

    Type getElementType();

    unsigned getNumElements();
}; // class PluginVectorType

class PluginFunctionType : public Type::TypeBase<PluginFunctionType, PluginTypeBase, detail::PluginFunctionTypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static bool isValidArgumentType(Type type);

    static bool isValidResultType(Type type);

    static PluginFunctionType get(MLIRContext *context, Type result, ArrayRef<Type> arguments);

    Type getReturnType();

    unsigned getNumParams();

    Type getParamType(unsigned i);

    ArrayRef<Type> getParams();

}; // class PluginFunctionType

class PluginStructType : public Type::TypeBase<PluginStructType, PluginTypeBase, detail::PluginStructTypeStorage> {
public:
    using Base::Base;

    PluginTypeID getPluginTypeID ();

    static bool isValidElementType(Type type);

    static PluginStructType get(MLIRContext *context, std::string name, ArrayRef<std::string> elemNames);

    std::string getName();

    ArrayRef<std::string> getElementNames();

}; // class PluginStructType

class PluginVoidType : public Type::TypeBase<PluginVoidType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;
    PluginTypeID getPluginTypeID ();
}; // class PluginVoidType

class PluginUndefType : public Type::TypeBase<PluginUndefType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;
    PluginTypeID getPluginTypeID ();
}; // class PluginUndefType

class PluginBooleanType : public Type::TypeBase<PluginBooleanType, PluginTypeBase, TypeStorage> {
public:
    using Base::Base;
    PluginTypeID getPluginTypeID ();
}; // class PluginBooleanType

} // namespace PluginIR

#endif // MLIR_DIALECT_PLUGINIR_PLUGINTYPES_H_