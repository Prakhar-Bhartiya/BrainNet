��	
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

: *
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
: *
dtype0
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

: @*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:@*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
��*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
J
0
1
2
3
'4
(5
/6
07
78
89*
J
0
1
2
3
'4
(5
/6
07
78
89*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Jserving_default* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEz_mean/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEz_mean/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEz_log_var/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEz_log_var/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 
* 
* 
* 
<
0
1
2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_4Placeholder*,
_output_shapes
:����������%*
dtype0*!
shape:����������%
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_33181
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_33417
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_33457՝
�q
�
B__inference_encoder_layer_call_and_return_conditional_losses_33150

inputs<
*dense_11_tensordot_readvariableop_resource: 6
(dense_11_biasadd_readvariableop_resource: <
*dense_12_tensordot_readvariableop_resource: @6
(dense_12_biasadd_readvariableop_resource:@;
'dense_13_matmul_readvariableop_resource:
��6
(dense_13_biasadd_readvariableop_resource:7
%z_mean_matmul_readvariableop_resource:4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense_11/BiasAdd/ReadVariableOp�!dense_11/Tensordot/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�!dense_12/Tensordot/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOp�
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0a
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_11/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_11/Tensordot/transpose	Transposeinputs"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������%�
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: b
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������% �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������% g
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*,
_output_shapes
:����������% �
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_12/Tensordot/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_12/Tensordot/transpose	Transposedense_11/Relu:activations:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������% �
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������%@�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������%@g
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*,
_output_shapes
:����������%@`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � �
flatten_1/ReshapeReshapedense_12/Relu:activations:0flatten_1/Const:output:0*
T0*)
_output_shapes
:������������
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_13/MatMulMatMulflatten_1/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:����������
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
z_mean/MatMulMatMuldense_13/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
z_log_var/MatMulMatMuldense_13/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������U
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2����
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:���������S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?z
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:���������u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:���������t
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:���������f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������%@:T P
,
_output_shapes
:����������%@
 
_user_specified_nameinputs
�
�
C__inference_dense_12_layer_call_and_return_conditional_losses_32518

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������% �
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������%@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������%@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������%@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������%@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������% 
 
_user_specified_nameinputs
�
�
(__inference_dense_13_layer_call_fn_33281

inputs
unknown:
��
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_32543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
&__inference_z_mean_layer_call_fn_33301

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_32559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_z_log_var_layer_call_fn_33320

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_33330

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
__inference__traced_save_33417
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*i
_input_shapesX
V: : : : @:@:
��:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:&"
 
_output_shapes
:
��: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: 
�	
�
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_11_layer_call_and_return_conditional_losses_32481

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������%�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������% r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������% U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������% f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������% z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�%
�
B__inference_encoder_layer_call_and_return_conditional_losses_32766

inputs 
dense_11_32736: 
dense_11_32738:  
dense_12_32741: @
dense_12_32743:@"
dense_13_32747:
��
dense_13_32749:
z_mean_32752:
z_mean_32754:!
z_log_var_32757:
z_log_var_32759:
identity

identity_1

identity_2�� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� sampling/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_32736dense_11_32738*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������% *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_32481�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_32741dense_12_32743*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_32518�
flatten_1/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_13_32747dense_13_32749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_32543�
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_mean_32752z_mean_32754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_32559�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_log_var_32757z_log_var_32759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575�
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_32607v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_32946

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_32766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�%
�
B__inference_encoder_layer_call_and_return_conditional_losses_32888
input_4 
dense_11_32858: 
dense_11_32860:  
dense_12_32863: @
dense_12_32865:@"
dense_13_32869:
��
dense_13_32871:
z_mean_32874:
z_mean_32876:!
z_log_var_32879:
z_log_var_32881:
identity

identity_1

identity_2�� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� sampling/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_11_32858dense_11_32860*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������% *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_32481�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_32863dense_12_32865*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_32518�
flatten_1/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_13_32869dense_13_32871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_32543�
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_mean_32874z_mean_32876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_32559�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_log_var_32879z_log_var_32881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575�
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_32607v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�
�
(__inference_dense_11_layer_call_fn_33190

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������% *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_32481t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������% `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������%: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�+
�
!__inference__traced_restore_33457
file_prefix2
 assignvariableop_dense_11_kernel: .
 assignvariableop_1_dense_11_bias: 4
"assignvariableop_2_dense_12_kernel: @.
 assignvariableop_3_dense_12_bias:@6
"assignvariableop_4_dense_13_kernel:
��.
 assignvariableop_5_dense_13_bias:2
 assignvariableop_6_z_mean_kernel:,
assignvariableop_7_z_mean_bias:5
#assignvariableop_8_z_log_var_kernel:/
!assignvariableop_9_z_log_var_bias:
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_12_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_12_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_13_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_13_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_z_mean_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_z_mean_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_z_log_var_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_z_log_var_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
r
C__inference_sampling_layer_call_and_return_conditional_losses_33362
inputs_0
inputs_1
identity�=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Shape_1Shapeinputs_0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2Ӎ��
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
q
(__inference_sampling_layer_call_fn_33336
inputs_0
inputs_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_32607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_33311

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
B__inference_encoder_layer_call_and_return_conditional_losses_32612

inputs 
dense_11_32482: 
dense_11_32484:  
dense_12_32519: @
dense_12_32521:@"
dense_13_32544:
��
dense_13_32546:
z_mean_32560:
z_mean_32562:!
z_log_var_32576:
z_log_var_32578:
identity

identity_1

identity_2�� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� sampling/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_32482dense_11_32484*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������% *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_32481�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_32519dense_12_32521*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_32518�
flatten_1/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_13_32544dense_13_32546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_32543�
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_mean_32560z_mean_32562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_32559�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_log_var_32576z_log_var_32578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575�
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_32607v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�
�
C__inference_dense_11_layer_call_and_return_conditional_losses_33221

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������%�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������% r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������% U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������% f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������% z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�
�
(__inference_dense_12_layer_call_fn_33230

inputs
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_32518t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������%@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������% : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������% 
 
_user_specified_nameinputs
�
�
'__inference_encoder_layer_call_fn_32822
input_4
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_32766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�
�
'__inference_encoder_layer_call_fn_32917

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_32612o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�

�
C__inference_dense_13_layer_call_and_return_conditional_losses_33292

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
p
C__inference_sampling_layer_call_and_return_conditional_losses_32607

inputs
inputs_1
identity�;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2����
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:���������E
ExpExpmul:z:0*
T0*'
_output_shapes
:���������Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:���������Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_12_layer_call_and_return_conditional_losses_33261

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:����������% �
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������%@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������%@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������%@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������%@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������% : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������% 
 
_user_specified_nameinputs
�%
�
B__inference_encoder_layer_call_and_return_conditional_losses_32855
input_4 
dense_11_32825: 
dense_11_32827:  
dense_12_32830: @
dense_12_32832:@"
dense_13_32836:
��
dense_13_32838:
z_mean_32841:
z_mean_32843:!
z_log_var_32846:
z_log_var_32848:
identity

identity_1

identity_2�� dense_11/StatefulPartitionedCall� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� sampling/StatefulPartitionedCall�!z_log_var/StatefulPartitionedCall�z_mean/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_11_32825dense_11_32827*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������% *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_32481�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall)dense_11/StatefulPartitionedCall:output:0dense_12_32830dense_12_32832*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������%@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_32518�
flatten_1/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_13_32836dense_13_32838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_32543�
z_mean/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_mean_32841z_mean_32843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_z_mean_layer_call_and_return_conditional_losses_32559�
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0z_log_var_32846z_log_var_32848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_z_log_var_layer_call_and_return_conditional_losses_32575�
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sampling_layer_call_and_return_conditional_losses_32607v
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������{

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������z

Identity_2Identity)sampling/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�

�
C__inference_dense_13_layer_call_and_return_conditional_losses_32543

inputs2
matmul_readvariableop_resource:
��-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_33272

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������%@:T P
,
_output_shapes
:����������%@
 
_user_specified_nameinputs
�q
�
B__inference_encoder_layer_call_and_return_conditional_losses_33048

inputs<
*dense_11_tensordot_readvariableop_resource: 6
(dense_11_biasadd_readvariableop_resource: <
*dense_12_tensordot_readvariableop_resource: @6
(dense_12_biasadd_readvariableop_resource:@;
'dense_13_matmul_readvariableop_resource:
��6
(dense_13_biasadd_readvariableop_resource:7
%z_mean_matmul_readvariableop_resource:4
&z_mean_biasadd_readvariableop_resource::
(z_log_var_matmul_readvariableop_resource:7
)z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��dense_11/BiasAdd/ReadVariableOp�!dense_11/Tensordot/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�!dense_12/Tensordot/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp� z_log_var/BiasAdd/ReadVariableOp�z_log_var/MatMul/ReadVariableOp�z_mean/BiasAdd/ReadVariableOp�z_mean/MatMul/ReadVariableOp�
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0a
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_11/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_11/Tensordot/transpose	Transposeinputs"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������%�
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: b
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������% �
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������% g
dense_11/ReluReludense_11/BiasAdd:output:0*
T0*,
_output_shapes
:����������% �
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_12/Tensordot/ShapeShapedense_11/Relu:activations:0*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_12/Tensordot/transpose	Transposedense_11/Relu:activations:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������% �
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@d
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@b
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������%@�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������%@g
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*,
_output_shapes
:����������%@`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � �
flatten_1/ReshapeReshapedense_12/Relu:activations:0flatten_1/Const:output:0*
T0*)
_output_shapes
:������������
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_13/MatMulMatMulflatten_1/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:����������
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
z_mean/MatMulMatMuldense_13/Relu:activations:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
z_log_var/MatMulMatMuldense_13/Relu:activations:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������U
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:f
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:h
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:`
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    b
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2����
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
sampling/random_normalAddV2sampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:���������S
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?z
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������W
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:���������u
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:���������t
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:���������f
IdentityIdentityz_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������k

Identity_1Identityz_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������a

Identity_2Identitysampling/add:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:T P
,
_output_shapes
:����������%
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_33181
input_4
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_32443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�
E
)__inference_flatten_1_layer_call_fn_33266

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_32530b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������%@:T P
,
_output_shapes
:����������%@
 
_user_specified_nameinputs
��
�	
 __inference__wrapped_model_32443
input_4D
2encoder_dense_11_tensordot_readvariableop_resource: >
0encoder_dense_11_biasadd_readvariableop_resource: D
2encoder_dense_12_tensordot_readvariableop_resource: @>
0encoder_dense_12_biasadd_readvariableop_resource:@C
/encoder_dense_13_matmul_readvariableop_resource:
��>
0encoder_dense_13_biasadd_readvariableop_resource:?
-encoder_z_mean_matmul_readvariableop_resource:<
.encoder_z_mean_biasadd_readvariableop_resource:B
0encoder_z_log_var_matmul_readvariableop_resource:?
1encoder_z_log_var_biasadd_readvariableop_resource:
identity

identity_1

identity_2��'encoder/dense_11/BiasAdd/ReadVariableOp�)encoder/dense_11/Tensordot/ReadVariableOp�'encoder/dense_12/BiasAdd/ReadVariableOp�)encoder/dense_12/Tensordot/ReadVariableOp�'encoder/dense_13/BiasAdd/ReadVariableOp�&encoder/dense_13/MatMul/ReadVariableOp�(encoder/z_log_var/BiasAdd/ReadVariableOp�'encoder/z_log_var/MatMul/ReadVariableOp�%encoder/z_mean/BiasAdd/ReadVariableOp�$encoder/z_mean/MatMul/ReadVariableOp�
)encoder/dense_11/Tensordot/ReadVariableOpReadVariableOp2encoder_dense_11_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0i
encoder/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
encoder/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       W
 encoder/dense_11/Tensordot/ShapeShapeinput_4*
T0*
_output_shapes
:j
(encoder/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#encoder/dense_11/Tensordot/GatherV2GatherV2)encoder/dense_11/Tensordot/Shape:output:0(encoder/dense_11/Tensordot/free:output:01encoder/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*encoder/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%encoder/dense_11/Tensordot/GatherV2_1GatherV2)encoder/dense_11/Tensordot/Shape:output:0(encoder/dense_11/Tensordot/axes:output:03encoder/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 encoder/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
encoder/dense_11/Tensordot/ProdProd,encoder/dense_11/Tensordot/GatherV2:output:0)encoder/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"encoder/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!encoder/dense_11/Tensordot/Prod_1Prod.encoder/dense_11/Tensordot/GatherV2_1:output:0+encoder/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&encoder/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!encoder/dense_11/Tensordot/concatConcatV2(encoder/dense_11/Tensordot/free:output:0(encoder/dense_11/Tensordot/axes:output:0/encoder/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 encoder/dense_11/Tensordot/stackPack(encoder/dense_11/Tensordot/Prod:output:0*encoder/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$encoder/dense_11/Tensordot/transpose	Transposeinput_4*encoder/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������%�
"encoder/dense_11/Tensordot/ReshapeReshape(encoder/dense_11/Tensordot/transpose:y:0)encoder/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!encoder/dense_11/Tensordot/MatMulMatMul+encoder/dense_11/Tensordot/Reshape:output:01encoder/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
"encoder/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: j
(encoder/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#encoder/dense_11/Tensordot/concat_1ConcatV2,encoder/dense_11/Tensordot/GatherV2:output:0+encoder/dense_11/Tensordot/Const_2:output:01encoder/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
encoder/dense_11/TensordotReshape+encoder/dense_11/Tensordot/MatMul:product:0,encoder/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������% �
'encoder/dense_11/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_11_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
encoder/dense_11/BiasAddBiasAdd#encoder/dense_11/Tensordot:output:0/encoder/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������% w
encoder/dense_11/ReluRelu!encoder/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:����������% �
)encoder/dense_12/Tensordot/ReadVariableOpReadVariableOp2encoder_dense_12_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0i
encoder/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:p
encoder/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       s
 encoder/dense_12/Tensordot/ShapeShape#encoder/dense_11/Relu:activations:0*
T0*
_output_shapes
:j
(encoder/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#encoder/dense_12/Tensordot/GatherV2GatherV2)encoder/dense_12/Tensordot/Shape:output:0(encoder/dense_12/Tensordot/free:output:01encoder/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*encoder/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%encoder/dense_12/Tensordot/GatherV2_1GatherV2)encoder/dense_12/Tensordot/Shape:output:0(encoder/dense_12/Tensordot/axes:output:03encoder/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 encoder/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
encoder/dense_12/Tensordot/ProdProd,encoder/dense_12/Tensordot/GatherV2:output:0)encoder/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"encoder/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!encoder/dense_12/Tensordot/Prod_1Prod.encoder/dense_12/Tensordot/GatherV2_1:output:0+encoder/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&encoder/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!encoder/dense_12/Tensordot/concatConcatV2(encoder/dense_12/Tensordot/free:output:0(encoder/dense_12/Tensordot/axes:output:0/encoder/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 encoder/dense_12/Tensordot/stackPack(encoder/dense_12/Tensordot/Prod:output:0*encoder/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$encoder/dense_12/Tensordot/transpose	Transpose#encoder/dense_11/Relu:activations:0*encoder/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������% �
"encoder/dense_12/Tensordot/ReshapeReshape(encoder/dense_12/Tensordot/transpose:y:0)encoder/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
!encoder/dense_12/Tensordot/MatMulMatMul+encoder/dense_12/Tensordot/Reshape:output:01encoder/dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
"encoder/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@j
(encoder/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#encoder/dense_12/Tensordot/concat_1ConcatV2,encoder/dense_12/Tensordot/GatherV2:output:0+encoder/dense_12/Tensordot/Const_2:output:01encoder/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
encoder/dense_12/TensordotReshape+encoder/dense_12/Tensordot/MatMul:product:0,encoder/dense_12/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������%@�
'encoder/dense_12/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_12_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
encoder/dense_12/BiasAddBiasAdd#encoder/dense_12/Tensordot:output:0/encoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������%@w
encoder/dense_12/ReluRelu!encoder/dense_12/BiasAdd:output:0*
T0*,
_output_shapes
:����������%@h
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� � �
encoder/flatten_1/ReshapeReshape#encoder/dense_12/Relu:activations:0 encoder/flatten_1/Const:output:0*
T0*)
_output_shapes
:������������
&encoder/dense_13/MatMul/ReadVariableOpReadVariableOp/encoder_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
encoder/dense_13/MatMulMatMul"encoder/flatten_1/Reshape:output:0.encoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'encoder/dense_13/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/dense_13/BiasAddBiasAdd!encoder/dense_13/MatMul:product:0/encoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
encoder/dense_13/ReluRelu!encoder/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder/z_mean/MatMulMatMul#encoder/dense_13/Relu:activations:0,encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/z_mean/BiasAddBiasAddencoder/z_mean/MatMul:product:0-encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
encoder/z_log_var/MatMulMatMul#encoder/dense_13/Relu:activations:0/encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
encoder/z_log_var/BiasAddBiasAdd"encoder/z_log_var/MatMul:product:00encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������e
encoder/sampling/ShapeShapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:n
$encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
encoder/sampling/strided_sliceStridedSliceencoder/sampling/Shape:output:0-encoder/sampling/strided_slice/stack:output:0/encoder/sampling/strided_slice/stack_1:output:0/encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
encoder/sampling/Shape_1Shapeencoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:p
&encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 encoder/sampling/strided_slice_1StridedSlice!encoder/sampling/Shape_1:output:0/encoder/sampling/strided_slice_1/stack:output:01encoder/sampling/strided_slice_1/stack_1:output:01encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
$encoder/sampling/random_normal/shapePack'encoder/sampling/strided_slice:output:0)encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:h
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    j
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-encoder/sampling/random_normal/shape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2�׃�
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:����������
encoder/sampling/random_normalAddV2&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:���������[
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
encoder/sampling/mulMulencoder/sampling/mul/x:output:0"encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:���������g
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:����������
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:����������
encoder/sampling/addAddV2encoder/z_mean/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:���������g
IdentityIdentityencoder/sampling/add:z:0^NoOp*
T0*'
_output_shapes
:���������s

Identity_1Identity"encoder/z_log_var/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������p

Identity_2Identityencoder/z_mean/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^encoder/dense_11/BiasAdd/ReadVariableOp*^encoder/dense_11/Tensordot/ReadVariableOp(^encoder/dense_12/BiasAdd/ReadVariableOp*^encoder/dense_12/Tensordot/ReadVariableOp(^encoder/dense_13/BiasAdd/ReadVariableOp'^encoder/dense_13/MatMul/ReadVariableOp)^encoder/z_log_var/BiasAdd/ReadVariableOp(^encoder/z_log_var/MatMul/ReadVariableOp&^encoder/z_mean/BiasAdd/ReadVariableOp%^encoder/z_mean/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 2R
'encoder/dense_11/BiasAdd/ReadVariableOp'encoder/dense_11/BiasAdd/ReadVariableOp2V
)encoder/dense_11/Tensordot/ReadVariableOp)encoder/dense_11/Tensordot/ReadVariableOp2R
'encoder/dense_12/BiasAdd/ReadVariableOp'encoder/dense_12/BiasAdd/ReadVariableOp2V
)encoder/dense_12/Tensordot/ReadVariableOp)encoder/dense_12/Tensordot/ReadVariableOp2R
'encoder/dense_13/BiasAdd/ReadVariableOp'encoder/dense_13/BiasAdd/ReadVariableOp2P
&encoder/dense_13/MatMul/ReadVariableOp&encoder/dense_13/MatMul/ReadVariableOp2T
(encoder/z_log_var/BiasAdd/ReadVariableOp(encoder/z_log_var/BiasAdd/ReadVariableOp2R
'encoder/z_log_var/MatMul/ReadVariableOp'encoder/z_log_var/MatMul/ReadVariableOp2N
%encoder/z_mean/BiasAdd/ReadVariableOp%encoder/z_mean/BiasAdd/ReadVariableOp2L
$encoder/z_mean/MatMul/ReadVariableOp$encoder/z_mean/MatMul/ReadVariableOp:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�
�
'__inference_encoder_layer_call_fn_32639
input_4
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:
��
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_32612o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������%: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������%
!
_user_specified_name	input_4
�	
�
A__inference_z_mean_layer_call_and_return_conditional_losses_32559

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_45
serving_default_input_4:0����������%<
sampling0
StatefulPartitionedCall:0���������=
	z_log_var0
StatefulPartitionedCall:1���������:
z_mean0
StatefulPartitionedCall:2���������tensorflow/serving/predict:�}
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
�

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
'4
(5
/6
07
78
89"
trackable_list_wrapper
f
0
1
2
3
'4
(5
/6
07
78
89"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_encoder_layer_call_fn_32639
'__inference_encoder_layer_call_fn_32917
'__inference_encoder_layer_call_fn_32946
'__inference_encoder_layer_call_fn_32822�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_encoder_layer_call_and_return_conditional_losses_33048
B__inference_encoder_layer_call_and_return_conditional_losses_33150
B__inference_encoder_layer_call_and_return_conditional_losses_32855
B__inference_encoder_layer_call_and_return_conditional_losses_32888�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_32443input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Jserving_default"
signature_map
!: 2dense_11/kernel
: 2dense_11/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_11_layer_call_fn_33190�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_11_layer_call_and_return_conditional_losses_33221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!: @2dense_12/kernel
:@2dense_12/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_12_layer_call_fn_33230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_12_layer_call_and_return_conditional_losses_33261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_flatten_1_layer_call_fn_33266�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_1_layer_call_and_return_conditional_losses_33272�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
#:!
��2dense_13/kernel
:2dense_13/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dense_13_layer_call_fn_33281�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_13_layer_call_and_return_conditional_losses_33292�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:2z_mean/kernel
:2z_mean/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�2�
&__inference_z_mean_layer_call_fn_33301�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_z_mean_layer_call_and_return_conditional_losses_33311�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
": 2z_log_var/kernel
:2z_log_var/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_z_log_var_layer_call_fn_33320�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_z_log_var_layer_call_and_return_conditional_losses_33330�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_sampling_layer_call_fn_33336�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_sampling_layer_call_and_return_conditional_losses_33362�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_signature_wrapper_33181input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
 __inference__wrapped_model_32443�
'(/0785�2
+�(
&�#
input_4����������%
� "���
.
sampling"�
sampling���������
0
	z_log_var#� 
	z_log_var���������
*
z_mean �
z_mean����������
C__inference_dense_11_layer_call_and_return_conditional_losses_33221f4�1
*�'
%�"
inputs����������%
� "*�'
 �
0����������% 
� �
(__inference_dense_11_layer_call_fn_33190Y4�1
*�'
%�"
inputs����������%
� "�����������% �
C__inference_dense_12_layer_call_and_return_conditional_losses_33261f4�1
*�'
%�"
inputs����������% 
� "*�'
 �
0����������%@
� �
(__inference_dense_12_layer_call_fn_33230Y4�1
*�'
%�"
inputs����������% 
� "�����������%@�
C__inference_dense_13_layer_call_and_return_conditional_losses_33292^'(1�.
'�$
"�
inputs�����������
� "%�"
�
0���������
� }
(__inference_dense_13_layer_call_fn_33281Q'(1�.
'�$
"�
inputs�����������
� "�����������
B__inference_encoder_layer_call_and_return_conditional_losses_32855�
'(/078=�:
3�0
&�#
input_4����������%
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_32888�
'(/078=�:
3�0
&�#
input_4����������%
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_33048�
'(/078<�9
2�/
%�"
inputs����������%
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_encoder_layer_call_and_return_conditional_losses_33150�
'(/078<�9
2�/
%�"
inputs����������%
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
'__inference_encoder_layer_call_fn_32639�
'(/078=�:
3�0
&�#
input_4����������%
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_32822�
'(/078=�:
3�0
&�#
input_4����������%
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_32917�
'(/078<�9
2�/
%�"
inputs����������%
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_encoder_layer_call_fn_32946�
'(/078<�9
2�/
%�"
inputs����������%
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
D__inference_flatten_1_layer_call_and_return_conditional_losses_33272_4�1
*�'
%�"
inputs����������%@
� "'�$
�
0�����������
� 
)__inference_flatten_1_layer_call_fn_33266R4�1
*�'
%�"
inputs����������%@
� "�������������
C__inference_sampling_layer_call_and_return_conditional_losses_33362�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
(__inference_sampling_layer_call_fn_33336vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
#__inference_signature_wrapper_33181�
'(/078@�=
� 
6�3
1
input_4&�#
input_4����������%"���
.
sampling"�
sampling���������
0
	z_log_var#� 
	z_log_var���������
*
z_mean �
z_mean����������
D__inference_z_log_var_layer_call_and_return_conditional_losses_33330\78/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� |
)__inference_z_log_var_layer_call_fn_33320O78/�,
%�"
 �
inputs���������
� "�����������
A__inference_z_mean_layer_call_and_return_conditional_losses_33311\/0/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� y
&__inference_z_mean_layer_call_fn_33301O/0/�,
%�"
 �
inputs���������
� "����������