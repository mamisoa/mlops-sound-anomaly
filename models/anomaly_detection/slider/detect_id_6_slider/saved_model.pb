��
��
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
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.22v2.8.2-0-g2ea19cbb5758��
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:*
dtype0
�
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_29/kernel
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*&
_output_shapes
:*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:*
dtype0
�
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_30/kernel
}
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*&
_output_shapes
:*
dtype0
t
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_30/bias
m
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes
:*
dtype0
�
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
:*
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
:*
dtype0
�
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:*
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
:*
dtype0
�
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:*
dtype0
�
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_28/kernel/m
�
+Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_28/bias/m
{
)Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_29/kernel/m
�
+Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_29/bias/m
{
)Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_30/kernel/m
�
+Adam/conv2d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_30/bias/m
{
)Adam/conv2d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_31/kernel/m
�
+Adam/conv2d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_31/bias/m
{
)Adam/conv2d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_32/kernel/m
�
+Adam/conv2d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_32/bias/m
{
)Adam/conv2d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_33/kernel/m
�
+Adam/conv2d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_33/bias/m
{
)Adam/conv2d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_34/kernel/m
�
+Adam/conv2d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/m*&
_output_shapes
:*
dtype0
�
Adam/conv2d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/m
{
)Adam/conv2d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_28/kernel/v
�
+Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_28/bias/v
{
)Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_29/kernel/v
�
+Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_29/bias/v
{
)Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_30/kernel/v
�
+Adam/conv2d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_30/bias/v
{
)Adam/conv2d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_31/kernel/v
�
+Adam/conv2d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_31/bias/v
{
)Adam/conv2d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_32/kernel/v
�
+Adam/conv2d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_32/bias/v
{
)Adam/conv2d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_33/kernel/v
�
+Adam/conv2d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_33/bias/v
{
)Adam/conv2d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv2d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_34/kernel/v
�
+Adam/conv2d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/v*&
_output_shapes
:*
dtype0
�
Adam/conv2d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/v
{
)Adam/conv2d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�s
value�sB�s B�s
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
�
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�*
* 
j
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813*
j
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

>serving_default* 
�

+kernel
,bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
�

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
�

/kernel
0bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
.
+0
,1
-2
.3
/4
05*
.
+0
,1
-2
.3
/4
05*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
�

1kernel
2bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
�

3kernel
4bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
�

5kernel
6bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�

7kernel
8bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
<
10
21
32
43
54
65
76
87*
<
10
21
32
43
54
65
76
87*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_28/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_29/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_29/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_30/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_30/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_31/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_31/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_32/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_32/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_33/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_33/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_34/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_34/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

�0
�1*
* 
* 
* 

+0
,1*

+0
,1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 
* 
* 

-0
.1*

-0
.1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses* 
* 
* 

/0
01*

/0
01*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 
* 
* 
* 
.
0
1
2
3
4
5*
* 
* 
* 

10
21*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 
* 
* 

30
41*

30
41*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 

50
61*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
5
0
1
2
3
4
5
6*
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
sm
VARIABLE_VALUEAdam/conv2d_28/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_28/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_29/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_29/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_30/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_30/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_31/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_31/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_32/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_32/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_33/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_33/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_34/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_34/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_28/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_28/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_29/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_29/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_30/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_30/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_31/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_31/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_32/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_32/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_33/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_33/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_34/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_34/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_imgPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_imgconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_3055137
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_28/kernel/m/Read/ReadVariableOp)Adam/conv2d_28/bias/m/Read/ReadVariableOp+Adam/conv2d_29/kernel/m/Read/ReadVariableOp)Adam/conv2d_29/bias/m/Read/ReadVariableOp+Adam/conv2d_30/kernel/m/Read/ReadVariableOp)Adam/conv2d_30/bias/m/Read/ReadVariableOp+Adam/conv2d_31/kernel/m/Read/ReadVariableOp)Adam/conv2d_31/bias/m/Read/ReadVariableOp+Adam/conv2d_32/kernel/m/Read/ReadVariableOp)Adam/conv2d_32/bias/m/Read/ReadVariableOp+Adam/conv2d_33/kernel/m/Read/ReadVariableOp)Adam/conv2d_33/bias/m/Read/ReadVariableOp+Adam/conv2d_34/kernel/m/Read/ReadVariableOp)Adam/conv2d_34/bias/m/Read/ReadVariableOp+Adam/conv2d_28/kernel/v/Read/ReadVariableOp)Adam/conv2d_28/bias/v/Read/ReadVariableOp+Adam/conv2d_29/kernel/v/Read/ReadVariableOp)Adam/conv2d_29/bias/v/Read/ReadVariableOp+Adam/conv2d_30/kernel/v/Read/ReadVariableOp)Adam/conv2d_30/bias/v/Read/ReadVariableOp+Adam/conv2d_31/kernel/v/Read/ReadVariableOp)Adam/conv2d_31/bias/v/Read/ReadVariableOp+Adam/conv2d_32/kernel/v/Read/ReadVariableOp)Adam/conv2d_32/bias/v/Read/ReadVariableOp+Adam/conv2d_33/kernel/v/Read/ReadVariableOp)Adam/conv2d_33/bias/v/Read/ReadVariableOp+Adam/conv2d_34/kernel/v/Read/ReadVariableOp)Adam/conv2d_34/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_3055754
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biastotalcounttotal_1count_1Adam/conv2d_28/kernel/mAdam/conv2d_28/bias/mAdam/conv2d_29/kernel/mAdam/conv2d_29/bias/mAdam/conv2d_30/kernel/mAdam/conv2d_30/bias/mAdam/conv2d_31/kernel/mAdam/conv2d_31/bias/mAdam/conv2d_32/kernel/mAdam/conv2d_32/bias/mAdam/conv2d_33/kernel/mAdam/conv2d_33/bias/mAdam/conv2d_34/kernel/mAdam/conv2d_34/bias/mAdam/conv2d_28/kernel/vAdam/conv2d_28/bias/vAdam/conv2d_29/kernel/vAdam/conv2d_29/bias/vAdam/conv2d_30/kernel/vAdam/conv2d_30/bias/vAdam/conv2d_31/kernel/vAdam/conv2d_31/bias/vAdam/conv2d_32/kernel/vAdam/conv2d_32/bias/vAdam/conv2d_33/kernel/vAdam/conv2d_33/bias/vAdam/conv2d_34/kernel/vAdam/conv2d_34/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_3055917��
�
�
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_12_layer_call_fn_3054826
img!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054762�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�}
�
"__inference__wrapped_model_3054025
img^
Dsequential_12_sequential_13_conv2d_28_conv2d_readvariableop_resource:S
Esequential_12_sequential_13_conv2d_28_biasadd_readvariableop_resource:^
Dsequential_12_sequential_13_conv2d_29_conv2d_readvariableop_resource:S
Esequential_12_sequential_13_conv2d_29_biasadd_readvariableop_resource:^
Dsequential_12_sequential_13_conv2d_30_conv2d_readvariableop_resource:S
Esequential_12_sequential_13_conv2d_30_biasadd_readvariableop_resource:^
Dsequential_12_sequential_14_conv2d_31_conv2d_readvariableop_resource:S
Esequential_12_sequential_14_conv2d_31_biasadd_readvariableop_resource:^
Dsequential_12_sequential_14_conv2d_32_conv2d_readvariableop_resource:S
Esequential_12_sequential_14_conv2d_32_biasadd_readvariableop_resource:^
Dsequential_12_sequential_14_conv2d_33_conv2d_readvariableop_resource:S
Esequential_12_sequential_14_conv2d_33_biasadd_readvariableop_resource:^
Dsequential_12_sequential_14_conv2d_34_conv2d_readvariableop_resource:S
Esequential_12_sequential_14_conv2d_34_biasadd_readvariableop_resource:
identity��<sequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOp�;sequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOp�<sequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOp�;sequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOp�<sequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOp�;sequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOp�<sequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOp�;sequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOp�<sequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOp�;sequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOp�<sequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOp�;sequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOp�<sequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOp�;sequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOp�
;sequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_13_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_13/conv2d_28/Conv2DConv2DimgCsequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
<sequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_13_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_13/conv2d_28/BiasAddBiasAdd5sequential_12/sequential_13/conv2d_28/Conv2D:output:0Dsequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
-sequential_12/sequential_13/conv2d_28/SigmoidSigmoid6sequential_12/sequential_13/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:������������
4sequential_12/sequential_13/max_pooling2d_12/MaxPoolMaxPool1sequential_12/sequential_13/conv2d_28/Sigmoid:y:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
;sequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_13_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_13/conv2d_29/Conv2DConv2D=sequential_12/sequential_13/max_pooling2d_12/MaxPool:output:0Csequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
<sequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_13_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_13/conv2d_29/BiasAddBiasAdd5sequential_12/sequential_13/conv2d_29/Conv2D:output:0Dsequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
*sequential_12/sequential_13/conv2d_29/ReluRelu6sequential_12/sequential_13/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
4sequential_12/sequential_13/max_pooling2d_13/MaxPoolMaxPool8sequential_12/sequential_13/conv2d_29/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
;sequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_13_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_13/conv2d_30/Conv2DConv2D=sequential_12/sequential_13/max_pooling2d_13/MaxPool:output:0Csequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
<sequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_13_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_13/conv2d_30/BiasAddBiasAdd5sequential_12/sequential_13/conv2d_30/Conv2D:output:0Dsequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
*sequential_12/sequential_13/conv2d_30/ReluRelu6sequential_12/sequential_13/conv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
4sequential_12/sequential_13/max_pooling2d_14/MaxPoolMaxPool8sequential_12/sequential_13/conv2d_30/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
;sequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_14_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_14/conv2d_31/Conv2DConv2D=sequential_12/sequential_13/max_pooling2d_14/MaxPool:output:0Csequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
<sequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_14_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_14/conv2d_31/BiasAddBiasAdd5sequential_12/sequential_14/conv2d_31/Conv2D:output:0Dsequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
*sequential_12/sequential_14/conv2d_31/ReluRelu6sequential_12/sequential_14/conv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:����������
2sequential_12/sequential_14/up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
4sequential_12/sequential_14/up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
0sequential_12/sequential_14/up_sampling2d_12/mulMul;sequential_12/sequential_14/up_sampling2d_12/Const:output:0=sequential_12/sequential_14/up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:�
Isequential_12/sequential_14/up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_12/sequential_14/conv2d_31/Relu:activations:04sequential_12/sequential_14/up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:���������88*
half_pixel_centers(�
;sequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_14_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_14/conv2d_32/Conv2DConv2DZsequential_12/sequential_14/up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0Csequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
<sequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_14_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_14/conv2d_32/BiasAddBiasAdd5sequential_12/sequential_14/conv2d_32/Conv2D:output:0Dsequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
*sequential_12/sequential_14/conv2d_32/ReluRelu6sequential_12/sequential_14/conv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
2sequential_12/sequential_14/up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   �
4sequential_12/sequential_14/up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
0sequential_12/sequential_14/up_sampling2d_13/mulMul;sequential_12/sequential_14/up_sampling2d_13/Const:output:0=sequential_12/sequential_14/up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:�
Isequential_12/sequential_14/up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_12/sequential_14/conv2d_32/Relu:activations:04sequential_12/sequential_14/up_sampling2d_13/mul:z:0*
T0*/
_output_shapes
:���������pp*
half_pixel_centers(�
;sequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_14_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_14/conv2d_33/Conv2DConv2DZsequential_12/sequential_14/up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0Csequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
<sequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_14_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_14/conv2d_33/BiasAddBiasAdd5sequential_12/sequential_14/conv2d_33/Conv2D:output:0Dsequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
*sequential_12/sequential_14/conv2d_33/ReluRelu6sequential_12/sequential_14/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
2sequential_12/sequential_14/up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   �
4sequential_12/sequential_14/up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
0sequential_12/sequential_14/up_sampling2d_14/mulMul;sequential_12/sequential_14/up_sampling2d_14/Const:output:0=sequential_12/sequential_14/up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:�
Isequential_12/sequential_14/up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_12/sequential_14/conv2d_33/Relu:activations:04sequential_12/sequential_14/up_sampling2d_14/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
;sequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOpReadVariableOpDsequential_12_sequential_14_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
,sequential_12/sequential_14/conv2d_34/Conv2DConv2DZsequential_12/sequential_14/up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0Csequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
<sequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOpReadVariableOpEsequential_12_sequential_14_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
-sequential_12/sequential_14/conv2d_34/BiasAddBiasAdd5sequential_12/sequential_14/conv2d_34/Conv2D:output:0Dsequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
-sequential_12/sequential_14/conv2d_34/SigmoidSigmoid6sequential_12/sequential_14/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:������������
IdentityIdentity1sequential_12/sequential_14/conv2d_34/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp=^sequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOp<^sequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOp=^sequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOp<^sequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOp=^sequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOp<^sequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOp=^sequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOp<^sequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOp=^sequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOp<^sequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOp=^sequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOp<^sequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOp=^sequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOp<^sequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2|
<sequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOp<sequential_12/sequential_13/conv2d_28/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOp;sequential_12/sequential_13/conv2d_28/Conv2D/ReadVariableOp2|
<sequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOp<sequential_12/sequential_13/conv2d_29/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOp;sequential_12/sequential_13/conv2d_29/Conv2D/ReadVariableOp2|
<sequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOp<sequential_12/sequential_13/conv2d_30/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOp;sequential_12/sequential_13/conv2d_30/Conv2D/ReadVariableOp2|
<sequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOp<sequential_12/sequential_14/conv2d_31/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOp;sequential_12/sequential_14/conv2d_31/Conv2D/ReadVariableOp2|
<sequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOp<sequential_12/sequential_14/conv2d_32/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOp;sequential_12/sequential_14/conv2d_32/Conv2D/ReadVariableOp2|
<sequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOp<sequential_12/sequential_14/conv2d_33/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOp;sequential_12/sequential_14/conv2d_33/Conv2D/ReadVariableOp2|
<sequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOp<sequential_12/sequential_14/conv2d_34/BiasAdd/ReadVariableOp2z
;sequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOp;sequential_12/sequential_14/conv2d_34/Conv2D/ReadVariableOp:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�#
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055227

inputsB
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:B
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:B
(conv2d_30_conv2d_readvariableop_resource:7
)conv2d_30_biasadd_readvariableop_resource:
identity�� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_28/SigmoidSigmoidconv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:������������
max_pooling2d_12/MaxPoolMaxPoolconv2d_28/Sigmoid:y:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_29/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppl
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
max_pooling2d_13/MaxPoolMaxPoolconv2d_29/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_30/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88l
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
max_pooling2d_14/MaxPoolMaxPoolconv2d_30/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity!max_pooling2d_14/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_30_layer_call_fn_3055426

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_3055137
img!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_3054025y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�	
�
/__inference_sequential_13_layer_call_fn_3054241
conv2d_28_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054209w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameconv2d_28_input
�
�
/__inference_sequential_12_layer_call_fn_3054933

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054662�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054285
conv2d_28_input+
conv2d_28_3054266:
conv2d_28_3054268:+
conv2d_29_3054272:
conv2d_29_3054274:+
conv2d_30_3054278:
conv2d_30_3054280:
identity��!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�!conv2d_30/StatefulPartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallconv2d_28_inputconv2d_28_3054266conv2d_28_3054268*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079�
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_29_3054272conv2d_29_3054274*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097�
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046�
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_30_3054278conv2d_30_3054280*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115�
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058�
IdentityIdentity)max_pooling2d_14/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameconv2d_28_input
�	
�
/__inference_sequential_13_layer_call_fn_3054138
conv2d_28_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_28_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054123w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameconv2d_28_input
�
�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
/__inference_sequential_13_layer_call_fn_3055154

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054123w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
/__inference_sequential_14_layer_call_fn_3055269

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054530�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3055377

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3055467

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3055504

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�5
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055313

inputsB
(conv2d_31_conv2d_readvariableop_resource:7
)conv2d_31_biasadd_readvariableop_resource:B
(conv2d_32_conv2d_readvariableop_resource:7
)conv2d_32_biasadd_readvariableop_resource:B
(conv2d_33_conv2d_readvariableop_resource:7
)conv2d_33_biasadd_readvariableop_resource:B
(conv2d_34_conv2d_readvariableop_resource:7
)conv2d_34_biasadd_readvariableop_resource:
identity�� conv2d_31/BiasAdd/ReadVariableOp�conv2d_31/Conv2D/ReadVariableOp� conv2d_32/BiasAdd/ReadVariableOp�conv2d_32/Conv2D/ReadVariableOp� conv2d_33/BiasAdd/ReadVariableOp�conv2d_33/Conv2D/ReadVariableOp� conv2d_34/BiasAdd/ReadVariableOp�conv2d_34/Conv2D/ReadVariableOp�
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:���������g
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_12/mulMulup_sampling2d_12/Const:output:0!up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_31/Relu:activations:0up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:���������88*
half_pixel_centers(�
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_32/Conv2DConv2D>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88l
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:���������88g
up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   i
up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_13/mulMulup_sampling2d_13/Const:output:0!up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_32/Relu:activations:0up_sampling2d_13/mul:z:0*
T0*/
_output_shapes
:���������pp*
half_pixel_centers(�
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_33/Conv2DConv2D>up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppl
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppg
up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   i
up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_14/mulMulup_sampling2d_14/Const:output:0!up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_33/Relu:activations:0up_sampling2d_14/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_34/Conv2DConv2D>up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_34/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054662

inputs/
sequential_13_3054631:#
sequential_13_3054633:/
sequential_13_3054635:#
sequential_13_3054637:/
sequential_13_3054639:#
sequential_13_3054641:/
sequential_14_3054644:#
sequential_14_3054646:/
sequential_14_3054648:#
sequential_14_3054650:/
sequential_14_3054652:#
sequential_14_3054654:/
sequential_14_3054656:#
sequential_14_3054658:
identity��%sequential_13/StatefulPartitionedCall�%sequential_14/StatefulPartitionedCall�
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_3054631sequential_13_3054633sequential_13_3054635sequential_13_3054637sequential_13_3054639sequential_13_3054641*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054123�
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_3054644sequential_14_3054646sequential_14_3054648sequential_14_3054650sequential_14_3054652sequential_14_3054654sequential_14_3054656sequential_14_3054658*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054421�
IdentityIdentity.sequential_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�"
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054624
conv2d_31_input+
conv2d_31_3054600:
conv2d_31_3054602:+
conv2d_32_3054606:
conv2d_32_3054608:+
conv2d_33_3054612:
conv2d_33_3054614:+
conv2d_34_3054618:
conv2d_34_3054620:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputconv2d_31_3054600conv2d_31_3054602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360�
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0conv2d_32_3054606conv2d_32_3054608*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378�
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_13/PartitionedCall:output:0conv2d_33_3054612conv2d_33_3054614*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396�
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_14/PartitionedCall:output:0conv2d_34_3054618conv2d_34_3054620*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414�
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_31_input
�
�
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�!
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054421

inputs+
conv2d_31_3054361:
conv2d_31_3054363:+
conv2d_32_3054379:
conv2d_32_3054381:+
conv2d_33_3054397:
conv2d_33_3054399:+
conv2d_34_3054415:
conv2d_34_3054417:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_3054361conv2d_31_3054363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360�
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0conv2d_32_3054379conv2d_32_3054381*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378�
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_13/PartitionedCall:output:0conv2d_33_3054397conv2d_33_3054399*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396�
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_14/PartitionedCall:output:0conv2d_34_3054415conv2d_34_3054417*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414�
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3055407

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������ppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�c
�
 __inference__traced_save_3055754
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_28_kernel_m_read_readvariableop4
0savev2_adam_conv2d_28_bias_m_read_readvariableop6
2savev2_adam_conv2d_29_kernel_m_read_readvariableop4
0savev2_adam_conv2d_29_bias_m_read_readvariableop6
2savev2_adam_conv2d_30_kernel_m_read_readvariableop4
0savev2_adam_conv2d_30_bias_m_read_readvariableop6
2savev2_adam_conv2d_31_kernel_m_read_readvariableop4
0savev2_adam_conv2d_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_32_kernel_m_read_readvariableop4
0savev2_adam_conv2d_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_33_kernel_m_read_readvariableop4
0savev2_adam_conv2d_33_bias_m_read_readvariableop6
2savev2_adam_conv2d_34_kernel_m_read_readvariableop4
0savev2_adam_conv2d_34_bias_m_read_readvariableop6
2savev2_adam_conv2d_28_kernel_v_read_readvariableop4
0savev2_adam_conv2d_28_bias_v_read_readvariableop6
2savev2_adam_conv2d_29_kernel_v_read_readvariableop4
0savev2_adam_conv2d_29_bias_v_read_readvariableop6
2savev2_adam_conv2d_30_kernel_v_read_readvariableop4
0savev2_adam_conv2d_30_bias_v_read_readvariableop6
2savev2_adam_conv2d_31_kernel_v_read_readvariableop4
0savev2_adam_conv2d_31_bias_v_read_readvariableop6
2savev2_adam_conv2d_32_kernel_v_read_readvariableop4
0savev2_adam_conv2d_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_33_kernel_v_read_readvariableop4
0savev2_adam_conv2d_33_bias_v_read_readvariableop6
2savev2_adam_conv2d_34_kernel_v_read_readvariableop4
0savev2_adam_conv2d_34_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_28_kernel_m_read_readvariableop0savev2_adam_conv2d_28_bias_m_read_readvariableop2savev2_adam_conv2d_29_kernel_m_read_readvariableop0savev2_adam_conv2d_29_bias_m_read_readvariableop2savev2_adam_conv2d_30_kernel_m_read_readvariableop0savev2_adam_conv2d_30_bias_m_read_readvariableop2savev2_adam_conv2d_31_kernel_m_read_readvariableop0savev2_adam_conv2d_31_bias_m_read_readvariableop2savev2_adam_conv2d_32_kernel_m_read_readvariableop0savev2_adam_conv2d_32_bias_m_read_readvariableop2savev2_adam_conv2d_33_kernel_m_read_readvariableop0savev2_adam_conv2d_33_bias_m_read_readvariableop2savev2_adam_conv2d_34_kernel_m_read_readvariableop0savev2_adam_conv2d_34_bias_m_read_readvariableop2savev2_adam_conv2d_28_kernel_v_read_readvariableop0savev2_adam_conv2d_28_bias_v_read_readvariableop2savev2_adam_conv2d_29_kernel_v_read_readvariableop0savev2_adam_conv2d_29_bias_v_read_readvariableop2savev2_adam_conv2d_30_kernel_v_read_readvariableop0savev2_adam_conv2d_30_bias_v_read_readvariableop2savev2_adam_conv2d_31_kernel_v_read_readvariableop0savev2_adam_conv2d_31_bias_v_read_readvariableop2savev2_adam_conv2d_32_kernel_v_read_readvariableop0savev2_adam_conv2d_32_bias_v_read_readvariableop2savev2_adam_conv2d_33_kernel_v_read_readvariableop0savev2_adam_conv2d_33_bias_v_read_readvariableop2savev2_adam_conv2d_34_kernel_v_read_readvariableop0savev2_adam_conv2d_34_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : ::::::::::::::: : : : ::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::4

_output_shapes
: 
�
i
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
i
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3055558

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������ppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������ppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054762

inputs/
sequential_13_3054731:#
sequential_13_3054733:/
sequential_13_3054735:#
sequential_13_3054737:/
sequential_13_3054739:#
sequential_13_3054741:/
sequential_14_3054744:#
sequential_14_3054746:/
sequential_14_3054748:#
sequential_14_3054750:/
sequential_14_3054752:#
sequential_14_3054754:/
sequential_14_3054756:#
sequential_14_3054758:
identity��%sequential_13/StatefulPartitionedCall�%sequential_14/StatefulPartitionedCall�
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_3054731sequential_13_3054733sequential_13_3054735sequential_13_3054737sequential_13_3054739sequential_13_3054741*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054209�
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_3054744sequential_14_3054746sequential_14_3054748sequential_14_3054750sequential_14_3054752sequential_14_3054754sequential_14_3054756sequential_14_3054758*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054530�
IdentityIdentity.sequential_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�i
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055034

inputsP
6sequential_13_conv2d_28_conv2d_readvariableop_resource:E
7sequential_13_conv2d_28_biasadd_readvariableop_resource:P
6sequential_13_conv2d_29_conv2d_readvariableop_resource:E
7sequential_13_conv2d_29_biasadd_readvariableop_resource:P
6sequential_13_conv2d_30_conv2d_readvariableop_resource:E
7sequential_13_conv2d_30_biasadd_readvariableop_resource:P
6sequential_14_conv2d_31_conv2d_readvariableop_resource:E
7sequential_14_conv2d_31_biasadd_readvariableop_resource:P
6sequential_14_conv2d_32_conv2d_readvariableop_resource:E
7sequential_14_conv2d_32_biasadd_readvariableop_resource:P
6sequential_14_conv2d_33_conv2d_readvariableop_resource:E
7sequential_14_conv2d_33_biasadd_readvariableop_resource:P
6sequential_14_conv2d_34_conv2d_readvariableop_resource:E
7sequential_14_conv2d_34_biasadd_readvariableop_resource:
identity��.sequential_13/conv2d_28/BiasAdd/ReadVariableOp�-sequential_13/conv2d_28/Conv2D/ReadVariableOp�.sequential_13/conv2d_29/BiasAdd/ReadVariableOp�-sequential_13/conv2d_29/Conv2D/ReadVariableOp�.sequential_13/conv2d_30/BiasAdd/ReadVariableOp�-sequential_13/conv2d_30/Conv2D/ReadVariableOp�.sequential_14/conv2d_31/BiasAdd/ReadVariableOp�-sequential_14/conv2d_31/Conv2D/ReadVariableOp�.sequential_14/conv2d_32/BiasAdd/ReadVariableOp�-sequential_14/conv2d_32/Conv2D/ReadVariableOp�.sequential_14/conv2d_33/BiasAdd/ReadVariableOp�-sequential_14/conv2d_33/Conv2D/ReadVariableOp�.sequential_14/conv2d_34/BiasAdd/ReadVariableOp�-sequential_14/conv2d_34/Conv2D/ReadVariableOp�
-sequential_13/conv2d_28/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_28/Conv2DConv2Dinputs5sequential_13/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
.sequential_13/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_28/BiasAddBiasAdd'sequential_13/conv2d_28/Conv2D:output:06sequential_13/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_13/conv2d_28/SigmoidSigmoid(sequential_13/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:������������
&sequential_13/max_pooling2d_12/MaxPoolMaxPool#sequential_13/conv2d_28/Sigmoid:y:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
-sequential_13/conv2d_29/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_29/Conv2DConv2D/sequential_13/max_pooling2d_12/MaxPool:output:05sequential_13/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
.sequential_13/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_29/BiasAddBiasAdd'sequential_13/conv2d_29/Conv2D:output:06sequential_13/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
sequential_13/conv2d_29/ReluRelu(sequential_13/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
&sequential_13/max_pooling2d_13/MaxPoolMaxPool*sequential_13/conv2d_29/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
-sequential_13/conv2d_30/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_30/Conv2DConv2D/sequential_13/max_pooling2d_13/MaxPool:output:05sequential_13/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
.sequential_13/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_30/BiasAddBiasAdd'sequential_13/conv2d_30/Conv2D:output:06sequential_13/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
sequential_13/conv2d_30/ReluRelu(sequential_13/conv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
&sequential_13/max_pooling2d_14/MaxPoolMaxPool*sequential_13/conv2d_30/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
-sequential_14/conv2d_31/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_31/Conv2DConv2D/sequential_13/max_pooling2d_14/MaxPool:output:05sequential_14/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_14/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_31/BiasAddBiasAdd'sequential_14/conv2d_31/Conv2D:output:06sequential_14/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_14/conv2d_31/ReluRelu(sequential_14/conv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:���������u
$sequential_14/up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      w
&sequential_14/up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_12/mulMul-sequential_14/up_sampling2d_12/Const:output:0/sequential_14/up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_31/Relu:activations:0&sequential_14/up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:���������88*
half_pixel_centers(�
-sequential_14/conv2d_32/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_32/Conv2DConv2DLsequential_14/up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
.sequential_14/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_32/BiasAddBiasAdd'sequential_14/conv2d_32/Conv2D:output:06sequential_14/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
sequential_14/conv2d_32/ReluRelu(sequential_14/conv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:���������88u
$sequential_14/up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   w
&sequential_14/up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_13/mulMul-sequential_14/up_sampling2d_13/Const:output:0/sequential_14/up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_32/Relu:activations:0&sequential_14/up_sampling2d_13/mul:z:0*
T0*/
_output_shapes
:���������pp*
half_pixel_centers(�
-sequential_14/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_33/Conv2DConv2DLsequential_14/up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
.sequential_14/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_33/BiasAddBiasAdd'sequential_14/conv2d_33/Conv2D:output:06sequential_14/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
sequential_14/conv2d_33/ReluRelu(sequential_14/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppu
$sequential_14/up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   w
&sequential_14/up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_14/mulMul-sequential_14/up_sampling2d_14/Const:output:0/sequential_14/up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_33/Relu:activations:0&sequential_14/up_sampling2d_14/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
-sequential_14/conv2d_34/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_34/Conv2DConv2DLsequential_14/up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
.sequential_14/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_34/BiasAddBiasAdd'sequential_14/conv2d_34/Conv2D:output:06sequential_14/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_14/conv2d_34/SigmoidSigmoid(sequential_14/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:�����������|
IdentityIdentity#sequential_14/conv2d_34/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp/^sequential_13/conv2d_28/BiasAdd/ReadVariableOp.^sequential_13/conv2d_28/Conv2D/ReadVariableOp/^sequential_13/conv2d_29/BiasAdd/ReadVariableOp.^sequential_13/conv2d_29/Conv2D/ReadVariableOp/^sequential_13/conv2d_30/BiasAdd/ReadVariableOp.^sequential_13/conv2d_30/Conv2D/ReadVariableOp/^sequential_14/conv2d_31/BiasAdd/ReadVariableOp.^sequential_14/conv2d_31/Conv2D/ReadVariableOp/^sequential_14/conv2d_32/BiasAdd/ReadVariableOp.^sequential_14/conv2d_32/Conv2D/ReadVariableOp/^sequential_14/conv2d_33/BiasAdd/ReadVariableOp.^sequential_14/conv2d_33/Conv2D/ReadVariableOp/^sequential_14/conv2d_34/BiasAdd/ReadVariableOp.^sequential_14/conv2d_34/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2`
.sequential_13/conv2d_28/BiasAdd/ReadVariableOp.sequential_13/conv2d_28/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_28/Conv2D/ReadVariableOp-sequential_13/conv2d_28/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_29/BiasAdd/ReadVariableOp.sequential_13/conv2d_29/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_29/Conv2D/ReadVariableOp-sequential_13/conv2d_29/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_30/BiasAdd/ReadVariableOp.sequential_13/conv2d_30/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_30/Conv2D/ReadVariableOp-sequential_13/conv2d_30/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_31/BiasAdd/ReadVariableOp.sequential_14/conv2d_31/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_31/Conv2D/ReadVariableOp-sequential_14/conv2d_31/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_32/BiasAdd/ReadVariableOp.sequential_14/conv2d_32/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_32/Conv2D/ReadVariableOp-sequential_14/conv2d_32/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_33/BiasAdd/ReadVariableOp.sequential_14/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_33/Conv2D/ReadVariableOp-sequential_14/conv2d_33/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_34/BiasAdd/ReadVariableOp.sequential_14/conv2d_34/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_34/Conv2D/ReadVariableOp-sequential_14/conv2d_34/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_34_layer_call_fn_3055567

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�i
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055102

inputsP
6sequential_13_conv2d_28_conv2d_readvariableop_resource:E
7sequential_13_conv2d_28_biasadd_readvariableop_resource:P
6sequential_13_conv2d_29_conv2d_readvariableop_resource:E
7sequential_13_conv2d_29_biasadd_readvariableop_resource:P
6sequential_13_conv2d_30_conv2d_readvariableop_resource:E
7sequential_13_conv2d_30_biasadd_readvariableop_resource:P
6sequential_14_conv2d_31_conv2d_readvariableop_resource:E
7sequential_14_conv2d_31_biasadd_readvariableop_resource:P
6sequential_14_conv2d_32_conv2d_readvariableop_resource:E
7sequential_14_conv2d_32_biasadd_readvariableop_resource:P
6sequential_14_conv2d_33_conv2d_readvariableop_resource:E
7sequential_14_conv2d_33_biasadd_readvariableop_resource:P
6sequential_14_conv2d_34_conv2d_readvariableop_resource:E
7sequential_14_conv2d_34_biasadd_readvariableop_resource:
identity��.sequential_13/conv2d_28/BiasAdd/ReadVariableOp�-sequential_13/conv2d_28/Conv2D/ReadVariableOp�.sequential_13/conv2d_29/BiasAdd/ReadVariableOp�-sequential_13/conv2d_29/Conv2D/ReadVariableOp�.sequential_13/conv2d_30/BiasAdd/ReadVariableOp�-sequential_13/conv2d_30/Conv2D/ReadVariableOp�.sequential_14/conv2d_31/BiasAdd/ReadVariableOp�-sequential_14/conv2d_31/Conv2D/ReadVariableOp�.sequential_14/conv2d_32/BiasAdd/ReadVariableOp�-sequential_14/conv2d_32/Conv2D/ReadVariableOp�.sequential_14/conv2d_33/BiasAdd/ReadVariableOp�-sequential_14/conv2d_33/Conv2D/ReadVariableOp�.sequential_14/conv2d_34/BiasAdd/ReadVariableOp�-sequential_14/conv2d_34/Conv2D/ReadVariableOp�
-sequential_13/conv2d_28/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_28/Conv2DConv2Dinputs5sequential_13/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
.sequential_13/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_28/BiasAddBiasAdd'sequential_13/conv2d_28/Conv2D:output:06sequential_13/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_13/conv2d_28/SigmoidSigmoid(sequential_13/conv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:������������
&sequential_13/max_pooling2d_12/MaxPoolMaxPool#sequential_13/conv2d_28/Sigmoid:y:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
-sequential_13/conv2d_29/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_29/Conv2DConv2D/sequential_13/max_pooling2d_12/MaxPool:output:05sequential_13/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
.sequential_13/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_29/BiasAddBiasAdd'sequential_13/conv2d_29/Conv2D:output:06sequential_13/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
sequential_13/conv2d_29/ReluRelu(sequential_13/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
&sequential_13/max_pooling2d_13/MaxPoolMaxPool*sequential_13/conv2d_29/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
-sequential_13/conv2d_30/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_13/conv2d_30/Conv2DConv2D/sequential_13/max_pooling2d_13/MaxPool:output:05sequential_13/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
.sequential_13/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_13/conv2d_30/BiasAddBiasAdd'sequential_13/conv2d_30/Conv2D:output:06sequential_13/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
sequential_13/conv2d_30/ReluRelu(sequential_13/conv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
&sequential_13/max_pooling2d_14/MaxPoolMaxPool*sequential_13/conv2d_30/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
�
-sequential_14/conv2d_31/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_31/Conv2DConv2D/sequential_13/max_pooling2d_14/MaxPool:output:05sequential_14/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_14/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_31/BiasAddBiasAdd'sequential_14/conv2d_31/Conv2D:output:06sequential_14/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_14/conv2d_31/ReluRelu(sequential_14/conv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:���������u
$sequential_14/up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      w
&sequential_14/up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_12/mulMul-sequential_14/up_sampling2d_12/Const:output:0/sequential_14/up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_31/Relu:activations:0&sequential_14/up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:���������88*
half_pixel_centers(�
-sequential_14/conv2d_32/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_32/Conv2DConv2DLsequential_14/up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
.sequential_14/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_32/BiasAddBiasAdd'sequential_14/conv2d_32/Conv2D:output:06sequential_14/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88�
sequential_14/conv2d_32/ReluRelu(sequential_14/conv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:���������88u
$sequential_14/up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   w
&sequential_14/up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_13/mulMul-sequential_14/up_sampling2d_13/Const:output:0/sequential_14/up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_32/Relu:activations:0&sequential_14/up_sampling2d_13/mul:z:0*
T0*/
_output_shapes
:���������pp*
half_pixel_centers(�
-sequential_14/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_33/Conv2DConv2DLsequential_14/up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
.sequential_14/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_33/BiasAddBiasAdd'sequential_14/conv2d_33/Conv2D:output:06sequential_14/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp�
sequential_14/conv2d_33/ReluRelu(sequential_14/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppu
$sequential_14/up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   w
&sequential_14/up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
"sequential_14/up_sampling2d_14/mulMul-sequential_14/up_sampling2d_14/Const:output:0/sequential_14/up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:�
;sequential_14/up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_14/conv2d_33/Relu:activations:0&sequential_14/up_sampling2d_14/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
-sequential_14/conv2d_34/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_14/conv2d_34/Conv2DConv2DLsequential_14/up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:05sequential_14/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
.sequential_14/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_14/conv2d_34/BiasAddBiasAdd'sequential_14/conv2d_34/Conv2D:output:06sequential_14/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
sequential_14/conv2d_34/SigmoidSigmoid(sequential_14/conv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:�����������|
IdentityIdentity#sequential_14/conv2d_34/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp/^sequential_13/conv2d_28/BiasAdd/ReadVariableOp.^sequential_13/conv2d_28/Conv2D/ReadVariableOp/^sequential_13/conv2d_29/BiasAdd/ReadVariableOp.^sequential_13/conv2d_29/Conv2D/ReadVariableOp/^sequential_13/conv2d_30/BiasAdd/ReadVariableOp.^sequential_13/conv2d_30/Conv2D/ReadVariableOp/^sequential_14/conv2d_31/BiasAdd/ReadVariableOp.^sequential_14/conv2d_31/Conv2D/ReadVariableOp/^sequential_14/conv2d_32/BiasAdd/ReadVariableOp.^sequential_14/conv2d_32/Conv2D/ReadVariableOp/^sequential_14/conv2d_33/BiasAdd/ReadVariableOp.^sequential_14/conv2d_33/Conv2D/ReadVariableOp/^sequential_14/conv2d_34/BiasAdd/ReadVariableOp.^sequential_14/conv2d_34/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2`
.sequential_13/conv2d_28/BiasAdd/ReadVariableOp.sequential_13/conv2d_28/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_28/Conv2D/ReadVariableOp-sequential_13/conv2d_28/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_29/BiasAdd/ReadVariableOp.sequential_13/conv2d_29/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_29/Conv2D/ReadVariableOp-sequential_13/conv2d_29/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_30/BiasAdd/ReadVariableOp.sequential_13/conv2d_30/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_30/Conv2D/ReadVariableOp-sequential_13/conv2d_30/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_31/BiasAdd/ReadVariableOp.sequential_14/conv2d_31/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_31/Conv2D/ReadVariableOp-sequential_14/conv2d_31/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_32/BiasAdd/ReadVariableOp.sequential_14/conv2d_32/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_32/Conv2D/ReadVariableOp-sequential_14/conv2d_32/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_33/BiasAdd/ReadVariableOp.sequential_14/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_33/Conv2D/ReadVariableOp-sequential_14/conv2d_33/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_34/BiasAdd/ReadVariableOp.sequential_14/conv2d_34/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_34/Conv2D/ReadVariableOp-sequential_14/conv2d_34/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3055484

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�!
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054530

inputs+
conv2d_31_3054506:
conv2d_31_3054508:+
conv2d_32_3054512:
conv2d_32_3054514:+
conv2d_33_3054518:
conv2d_33_3054520:+
conv2d_34_3054524:
conv2d_34_3054526:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_3054506conv2d_31_3054508*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360�
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0conv2d_32_3054512conv2d_32_3054514*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378�
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_13/PartitionedCall:output:0conv2d_33_3054518conv2d_33_3054520*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396�
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_14/PartitionedCall:output:0conv2d_34_3054524conv2d_34_3054526*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414�
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_sequential_12_layer_call_fn_3054966

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054762�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054263
conv2d_28_input+
conv2d_28_3054244:
conv2d_28_3054246:+
conv2d_29_3054250:
conv2d_29_3054252:+
conv2d_30_3054256:
conv2d_30_3054258:
identity��!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�!conv2d_30/StatefulPartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallconv2d_28_inputconv2d_28_3054244conv2d_28_3054246*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079�
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_29_3054250conv2d_29_3054252*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097�
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046�
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_30_3054256conv2d_30_3054258*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115�
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058�
IdentityIdentity)max_pooling2d_14/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nameconv2d_28_input
�
N
2__inference_max_pooling2d_14_layer_call_fn_3055442

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
/__inference_sequential_14_layer_call_fn_3054440
conv2d_31_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054421�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_31_input
�

�
/__inference_sequential_14_layer_call_fn_3054570
conv2d_31_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054530�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_31_input
�
i
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3055521

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3055578

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
��
� 
#__inference__traced_restore_3055917
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: =
#assignvariableop_5_conv2d_28_kernel:/
!assignvariableop_6_conv2d_28_bias:=
#assignvariableop_7_conv2d_29_kernel:/
!assignvariableop_8_conv2d_29_bias:=
#assignvariableop_9_conv2d_30_kernel:0
"assignvariableop_10_conv2d_30_bias:>
$assignvariableop_11_conv2d_31_kernel:0
"assignvariableop_12_conv2d_31_bias:>
$assignvariableop_13_conv2d_32_kernel:0
"assignvariableop_14_conv2d_32_bias:>
$assignvariableop_15_conv2d_33_kernel:0
"assignvariableop_16_conv2d_33_bias:>
$assignvariableop_17_conv2d_34_kernel:0
"assignvariableop_18_conv2d_34_bias:#
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: E
+assignvariableop_23_adam_conv2d_28_kernel_m:7
)assignvariableop_24_adam_conv2d_28_bias_m:E
+assignvariableop_25_adam_conv2d_29_kernel_m:7
)assignvariableop_26_adam_conv2d_29_bias_m:E
+assignvariableop_27_adam_conv2d_30_kernel_m:7
)assignvariableop_28_adam_conv2d_30_bias_m:E
+assignvariableop_29_adam_conv2d_31_kernel_m:7
)assignvariableop_30_adam_conv2d_31_bias_m:E
+assignvariableop_31_adam_conv2d_32_kernel_m:7
)assignvariableop_32_adam_conv2d_32_bias_m:E
+assignvariableop_33_adam_conv2d_33_kernel_m:7
)assignvariableop_34_adam_conv2d_33_bias_m:E
+assignvariableop_35_adam_conv2d_34_kernel_m:7
)assignvariableop_36_adam_conv2d_34_bias_m:E
+assignvariableop_37_adam_conv2d_28_kernel_v:7
)assignvariableop_38_adam_conv2d_28_bias_v:E
+assignvariableop_39_adam_conv2d_29_kernel_v:7
)assignvariableop_40_adam_conv2d_29_bias_v:E
+assignvariableop_41_adam_conv2d_30_kernel_v:7
)assignvariableop_42_adam_conv2d_30_bias_v:E
+assignvariableop_43_adam_conv2d_31_kernel_v:7
)assignvariableop_44_adam_conv2d_31_bias_v:E
+assignvariableop_45_adam_conv2d_32_kernel_v:7
)assignvariableop_46_adam_conv2d_32_bias_v:E
+assignvariableop_47_adam_conv2d_33_kernel_v:7
)assignvariableop_48_adam_conv2d_33_bias_v:E
+assignvariableop_49_adam_conv2d_34_kernel_v:7
)assignvariableop_50_adam_conv2d_34_bias_v:
identity_52��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_28_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_28_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_29_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_29_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_30_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_30_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_31_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_31_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_32_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_32_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_33_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_33_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_34_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_34_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_28_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_28_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_29_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_29_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_30_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_30_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_31_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_31_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_32_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_32_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_33_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_33_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_34_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_34_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_28_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_28_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_29_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_29_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_30_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_30_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_31_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_31_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_32_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_32_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_33_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_33_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_34_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_34_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054894
img/
sequential_13_3054863:#
sequential_13_3054865:/
sequential_13_3054867:#
sequential_13_3054869:/
sequential_13_3054871:#
sequential_13_3054873:/
sequential_14_3054876:#
sequential_14_3054878:/
sequential_14_3054880:#
sequential_14_3054882:/
sequential_14_3054884:#
sequential_14_3054886:/
sequential_14_3054888:#
sequential_14_3054890:
identity��%sequential_13/StatefulPartitionedCall�%sequential_14/StatefulPartitionedCall�
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallimgsequential_13_3054863sequential_13_3054865sequential_13_3054867sequential_13_3054869sequential_13_3054871sequential_13_3054873*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054209�
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_3054876sequential_14_3054878sequential_14_3054880sequential_14_3054882sequential_14_3054884sequential_14_3054886sequential_14_3054888sequential_14_3054890*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054530�
IdentityIdentity.sequential_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�5
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055357

inputsB
(conv2d_31_conv2d_readvariableop_resource:7
)conv2d_31_biasadd_readvariableop_resource:B
(conv2d_32_conv2d_readvariableop_resource:7
)conv2d_32_biasadd_readvariableop_resource:B
(conv2d_33_conv2d_readvariableop_resource:7
)conv2d_33_biasadd_readvariableop_resource:B
(conv2d_34_conv2d_readvariableop_resource:7
)conv2d_34_biasadd_readvariableop_resource:
identity�� conv2d_31/BiasAdd/ReadVariableOp�conv2d_31/Conv2D/ReadVariableOp� conv2d_32/BiasAdd/ReadVariableOp�conv2d_32/Conv2D/ReadVariableOp� conv2d_33/BiasAdd/ReadVariableOp�conv2d_33/Conv2D/ReadVariableOp� conv2d_34/BiasAdd/ReadVariableOp�conv2d_34/Conv2D/ReadVariableOp�
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������l
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:���������g
up_sampling2d_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_12/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_12/mulMulup_sampling2d_12/Const:output:0!up_sampling2d_12/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_12/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_31/Relu:activations:0up_sampling2d_12/mul:z:0*
T0*/
_output_shapes
:���������88*
half_pixel_centers(�
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_32/Conv2DConv2D>up_sampling2d_12/resize/ResizeNearestNeighbor:resized_images:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88l
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:���������88g
up_sampling2d_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   i
up_sampling2d_13/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_13/mulMulup_sampling2d_13/Const:output:0!up_sampling2d_13/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_13/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_32/Relu:activations:0up_sampling2d_13/mul:z:0*
T0*/
_output_shapes
:���������pp*
half_pixel_centers(�
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_33/Conv2DConv2D>up_sampling2d_13/resize/ResizeNearestNeighbor:resized_images:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppl
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������ppg
up_sampling2d_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   i
up_sampling2d_14/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_14/mulMulup_sampling2d_14/Const:output:0!up_sampling2d_14/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_14/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_33/Relu:activations:0up_sampling2d_14/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_34/Conv2DConv2D>up_sampling2d_14/resize/ResizeNearestNeighbor:resized_images:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_34/SigmoidSigmoidconv2d_34/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_34/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_12_layer_call_fn_3055382

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054209

inputs+
conv2d_28_3054190:
conv2d_28_3054192:+
conv2d_29_3054196:
conv2d_29_3054198:+
conv2d_30_3054202:
conv2d_30_3054204:
identity��!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�!conv2d_30/StatefulPartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_3054190conv2d_28_3054192*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079�
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_29_3054196conv2d_29_3054198*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097�
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046�
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_30_3054202conv2d_30_3054204*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115�
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058�
IdentityIdentity)max_pooling2d_14/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3055417

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
N
2__inference_max_pooling2d_13_layer_call_fn_3055412

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
N
2__inference_up_sampling2d_13_layer_call_fn_3055509

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
N
2__inference_up_sampling2d_12_layer_call_fn_3055472

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�#
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055199

inputsB
(conv2d_28_conv2d_readvariableop_resource:7
)conv2d_28_biasadd_readvariableop_resource:B
(conv2d_29_conv2d_readvariableop_resource:7
)conv2d_29_biasadd_readvariableop_resource:B
(conv2d_30_conv2d_readvariableop_resource:7
)conv2d_30_biasadd_readvariableop_resource:
identity�� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_28/Conv2DConv2Dinputs'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_28/SigmoidSigmoidconv2d_28/BiasAdd:output:0*
T0*1
_output_shapes
:������������
max_pooling2d_12/MaxPoolMaxPoolconv2d_28/Sigmoid:y:0*/
_output_shapes
:���������pp*
ksize
*
paddingVALID*
strides
�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_29/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������ppl
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp�
max_pooling2d_13/MaxPoolMaxPoolconv2d_29/Relu:activations:0*/
_output_shapes
:���������88*
ksize
*
paddingVALID*
strides
�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_30/Conv2DConv2D!max_pooling2d_13/MaxPool:output:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88l
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:���������88�
max_pooling2d_14/MaxPoolMaxPoolconv2d_30/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity!max_pooling2d_14/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
/__inference_sequential_12_layer_call_fn_3054693
img!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054662�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�
�
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3055437

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������88
 
_user_specified_nameinputs
�
�
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054860
img/
sequential_13_3054829:#
sequential_13_3054831:/
sequential_13_3054833:#
sequential_13_3054835:/
sequential_13_3054837:#
sequential_13_3054839:/
sequential_14_3054842:#
sequential_14_3054844:/
sequential_14_3054846:#
sequential_14_3054848:/
sequential_14_3054850:#
sequential_14_3054852:/
sequential_14_3054854:#
sequential_14_3054856:
identity��%sequential_13/StatefulPartitionedCall�%sequential_14/StatefulPartitionedCall�
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallimgsequential_13_3054829sequential_13_3054831sequential_13_3054833sequential_13_3054835sequential_13_3054837sequential_13_3054839*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054123�
%sequential_14/StatefulPartitionedCallStatefulPartitionedCall.sequential_13/StatefulPartitionedCall:output:0sequential_14_3054842sequential_14_3054844sequential_14_3054846sequential_14_3054848sequential_14_3054850sequential_14_3054852sequential_14_3054854sequential_14_3054856*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054421�
IdentityIdentity.sequential_14/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp&^sequential_13/StatefulPartitionedCall&^sequential_14/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:�����������: : : : : : : : : : : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall2N
%sequential_14/StatefulPartitionedCall%sequential_14/StatefulPartitionedCall:V R
1
_output_shapes
:�����������

_user_specified_nameimg
�
�
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054123

inputs+
conv2d_28_3054080:
conv2d_28_3054082:+
conv2d_29_3054098:
conv2d_29_3054100:+
conv2d_30_3054116:
conv2d_30_3054118:
identity��!conv2d_28/StatefulPartitionedCall�!conv2d_29/StatefulPartitionedCall�!conv2d_30/StatefulPartitionedCall�
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_28_3054080conv2d_28_3054082*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079�
 max_pooling2d_12/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3054034�
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_29_3054098conv2d_29_3054100*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097�
 max_pooling2d_13/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3054046�
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_13/PartitionedCall:output:0conv2d_30_3054116conv2d_30_3054118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3054115�
 max_pooling2d_14/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058�
IdentityIdentity)max_pooling2d_14/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
/__inference_sequential_13_layer_call_fn_3055171

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054209w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):�����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_32_layer_call_fn_3055493

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3055447

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�"
�
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054597
conv2d_31_input+
conv2d_31_3054573:
conv2d_31_3054575:+
conv2d_32_3054579:
conv2d_32_3054581:+
conv2d_33_3054585:
conv2d_33_3054587:+
conv2d_34_3054591:
conv2d_34_3054593:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputconv2d_31_3054573conv2d_31_3054575*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360�
 up_sampling2d_12/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3054301�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_12/PartitionedCall:output:0conv2d_32_3054579conv2d_32_3054581*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3054378�
 up_sampling2d_13/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3054320�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_13/PartitionedCall:output:0conv2d_33_3054585conv2d_33_3054587*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396�
 up_sampling2d_14/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_14/PartitionedCall:output:0conv2d_34_3054591conv2d_34_3054593*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3054414�
IdentityIdentity*conv2d_34/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall:` \
/
_output_shapes
:���������
)
_user_specified_nameconv2d_31_input
�
�
+__inference_conv2d_29_layer_call_fn_3055396

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3054097w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������pp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������pp
 
_user_specified_nameinputs
�

�
/__inference_sequential_14_layer_call_fn_3055248

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054421�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_28_layer_call_fn_3055366

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3054079y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
N
2__inference_up_sampling2d_14_layer_call_fn_3055546

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3054339�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_31_layer_call_fn_3055456

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3054360w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3055541

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_33_layer_call_fn_3055530

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3054396�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3054058

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
i
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3055387

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
img6
serving_default_img:0�����������K
sequential_14:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�6m�7m�8m�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�6v�7v�8v�"
	optimizer
 "
trackable_list_wrapper
�
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813"
trackable_list_wrapper
�
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_sequential_12_layer_call_fn_3054693
/__inference_sequential_12_layer_call_fn_3054933
/__inference_sequential_12_layer_call_fn_3054966
/__inference_sequential_12_layer_call_fn_3054826�
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055034
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055102
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054860
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054894�
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
"__inference__wrapped_model_3054025img"�
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
>serving_default"
signature_map
�

+kernel
,bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
�
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
�

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
�

/kernel
0bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
J
+0
,1
-2
.3
/4
05"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_sequential_13_layer_call_fn_3054138
/__inference_sequential_13_layer_call_fn_3055154
/__inference_sequential_13_layer_call_fn_3055171
/__inference_sequential_13_layer_call_fn_3054241�
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055199
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055227
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054263
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054285�
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
�

1kernel
2bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
�

3kernel
4bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

5kernel
6bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

7kernel
8bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
X
10
21
32
43
54
65
76
87"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_sequential_14_layer_call_fn_3054440
/__inference_sequential_14_layer_call_fn_3055248
/__inference_sequential_14_layer_call_fn_3055269
/__inference_sequential_14_layer_call_fn_3054570�
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
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055313
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055357
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054597
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054624�
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv2d_28/kernel
:2conv2d_28/bias
*:(2conv2d_29/kernel
:2conv2d_29/bias
*:(2conv2d_30/kernel
:2conv2d_30/bias
*:(2conv2d_31/kernel
:2conv2d_31/bias
*:(2conv2d_32/kernel
:2conv2d_32/bias
*:(2conv2d_33/kernel
:2conv2d_33/bias
*:(2conv2d_34/kernel
:2conv2d_34/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_signature_wrapper_3055137img"�
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
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_28_layer_call_fn_3055366�
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
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3055377�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_max_pooling2d_12_layer_call_fn_3055382�
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
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3055387�
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
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_29_layer_call_fn_3055396�
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
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3055407�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_max_pooling2d_13_layer_call_fn_3055412�
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
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3055417�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_30_layer_call_fn_3055426�
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
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3055437�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_max_pooling2d_14_layer_call_fn_3055442�
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
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3055447�
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
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_31_layer_call_fn_3055456�
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
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3055467�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_up_sampling2d_12_layer_call_fn_3055472�
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
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3055484�
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
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_32_layer_call_fn_3055493�
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
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3055504�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_up_sampling2d_13_layer_call_fn_3055509�
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
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3055521�
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
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_33_layer_call_fn_3055530�
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
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3055541�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_up_sampling2d_14_layer_call_fn_3055546�
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
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3055558�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_34_layer_call_fn_3055567�
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
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3055578�
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
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
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
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
/:-2Adam/conv2d_28/kernel/m
!:2Adam/conv2d_28/bias/m
/:-2Adam/conv2d_29/kernel/m
!:2Adam/conv2d_29/bias/m
/:-2Adam/conv2d_30/kernel/m
!:2Adam/conv2d_30/bias/m
/:-2Adam/conv2d_31/kernel/m
!:2Adam/conv2d_31/bias/m
/:-2Adam/conv2d_32/kernel/m
!:2Adam/conv2d_32/bias/m
/:-2Adam/conv2d_33/kernel/m
!:2Adam/conv2d_33/bias/m
/:-2Adam/conv2d_34/kernel/m
!:2Adam/conv2d_34/bias/m
/:-2Adam/conv2d_28/kernel/v
!:2Adam/conv2d_28/bias/v
/:-2Adam/conv2d_29/kernel/v
!:2Adam/conv2d_29/bias/v
/:-2Adam/conv2d_30/kernel/v
!:2Adam/conv2d_30/bias/v
/:-2Adam/conv2d_31/kernel/v
!:2Adam/conv2d_31/bias/v
/:-2Adam/conv2d_32/kernel/v
!:2Adam/conv2d_32/bias/v
/:-2Adam/conv2d_33/kernel/v
!:2Adam/conv2d_33/bias/v
/:-2Adam/conv2d_34/kernel/v
!:2Adam/conv2d_34/bias/v�
"__inference__wrapped_model_3054025�+,-./0123456786�3
,�)
'�$
img�����������
� "G�D
B
sequential_141�.
sequential_14������������
F__inference_conv2d_28_layer_call_and_return_conditional_losses_3055377p+,9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
+__inference_conv2d_28_layer_call_fn_3055366c+,9�6
/�,
*�'
inputs�����������
� ""�������������
F__inference_conv2d_29_layer_call_and_return_conditional_losses_3055407l-.7�4
-�*
(�%
inputs���������pp
� "-�*
#� 
0���������pp
� �
+__inference_conv2d_29_layer_call_fn_3055396_-.7�4
-�*
(�%
inputs���������pp
� " ����������pp�
F__inference_conv2d_30_layer_call_and_return_conditional_losses_3055437l/07�4
-�*
(�%
inputs���������88
� "-�*
#� 
0���������88
� �
+__inference_conv2d_30_layer_call_fn_3055426_/07�4
-�*
(�%
inputs���������88
� " ����������88�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_3055467l127�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
+__inference_conv2d_31_layer_call_fn_3055456_127�4
-�*
(�%
inputs���������
� " �����������
F__inference_conv2d_32_layer_call_and_return_conditional_losses_3055504�34I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
+__inference_conv2d_32_layer_call_fn_3055493�34I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
F__inference_conv2d_33_layer_call_and_return_conditional_losses_3055541�56I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
+__inference_conv2d_33_layer_call_fn_3055530�56I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
F__inference_conv2d_34_layer_call_and_return_conditional_losses_3055578�78I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
+__inference_conv2d_34_layer_call_fn_3055567�78I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
M__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_3055387�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_12_layer_call_fn_3055382�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_3055417�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_13_layer_call_fn_3055412�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_max_pooling2d_14_layer_call_and_return_conditional_losses_3055447�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_max_pooling2d_14_layer_call_fn_3055442�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054860�+,-./012345678>�;
4�1
'�$
img�����������
p 

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_12_layer_call_and_return_conditional_losses_3054894�+,-./012345678>�;
4�1
'�$
img�����������
p

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055034�+,-./012345678A�>
7�4
*�'
inputs�����������
p 

 
� "/�,
%�"
0�����������
� �
J__inference_sequential_12_layer_call_and_return_conditional_losses_3055102�+,-./012345678A�>
7�4
*�'
inputs�����������
p

 
� "/�,
%�"
0�����������
� �
/__inference_sequential_12_layer_call_fn_3054693�+,-./012345678>�;
4�1
'�$
img�����������
p 

 
� "2�/+����������������������������
/__inference_sequential_12_layer_call_fn_3054826�+,-./012345678>�;
4�1
'�$
img�����������
p

 
� "2�/+����������������������������
/__inference_sequential_12_layer_call_fn_3054933�+,-./012345678A�>
7�4
*�'
inputs�����������
p 

 
� "2�/+����������������������������
/__inference_sequential_12_layer_call_fn_3054966�+,-./012345678A�>
7�4
*�'
inputs�����������
p

 
� "2�/+����������������������������
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054263�+,-./0J�G
@�=
3�0
conv2d_28_input�����������
p 

 
� "-�*
#� 
0���������
� �
J__inference_sequential_13_layer_call_and_return_conditional_losses_3054285�+,-./0J�G
@�=
3�0
conv2d_28_input�����������
p

 
� "-�*
#� 
0���������
� �
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055199z+,-./0A�>
7�4
*�'
inputs�����������
p 

 
� "-�*
#� 
0���������
� �
J__inference_sequential_13_layer_call_and_return_conditional_losses_3055227z+,-./0A�>
7�4
*�'
inputs�����������
p

 
� "-�*
#� 
0���������
� �
/__inference_sequential_13_layer_call_fn_3054138v+,-./0J�G
@�=
3�0
conv2d_28_input�����������
p 

 
� " �����������
/__inference_sequential_13_layer_call_fn_3054241v+,-./0J�G
@�=
3�0
conv2d_28_input�����������
p

 
� " �����������
/__inference_sequential_13_layer_call_fn_3055154m+,-./0A�>
7�4
*�'
inputs�����������
p 

 
� " �����������
/__inference_sequential_13_layer_call_fn_3055171m+,-./0A�>
7�4
*�'
inputs�����������
p

 
� " �����������
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054597�12345678H�E
>�;
1�.
conv2d_31_input���������
p 

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_14_layer_call_and_return_conditional_losses_3054624�12345678H�E
>�;
1�.
conv2d_31_input���������
p

 
� "?�<
5�2
0+���������������������������
� �
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055313|12345678?�<
5�2
(�%
inputs���������
p 

 
� "/�,
%�"
0�����������
� �
J__inference_sequential_14_layer_call_and_return_conditional_losses_3055357|12345678?�<
5�2
(�%
inputs���������
p

 
� "/�,
%�"
0�����������
� �
/__inference_sequential_14_layer_call_fn_3054440�12345678H�E
>�;
1�.
conv2d_31_input���������
p 

 
� "2�/+����������������������������
/__inference_sequential_14_layer_call_fn_3054570�12345678H�E
>�;
1�.
conv2d_31_input���������
p

 
� "2�/+����������������������������
/__inference_sequential_14_layer_call_fn_305524812345678?�<
5�2
(�%
inputs���������
p 

 
� "2�/+����������������������������
/__inference_sequential_14_layer_call_fn_305526912345678?�<
5�2
(�%
inputs���������
p

 
� "2�/+����������������������������
%__inference_signature_wrapper_3055137�+,-./012345678=�:
� 
3�0
.
img'�$
img�����������"G�D
B
sequential_141�.
sequential_14������������
M__inference_up_sampling2d_12_layer_call_and_return_conditional_losses_3055484�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_up_sampling2d_12_layer_call_fn_3055472�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_up_sampling2d_13_layer_call_and_return_conditional_losses_3055521�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_up_sampling2d_13_layer_call_fn_3055509�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
M__inference_up_sampling2d_14_layer_call_and_return_conditional_losses_3055558�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_up_sampling2d_14_layer_call_fn_3055546�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������