þ
á
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	

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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758æ
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

conv2d_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_84/kernel
}
$conv2d_84/kernel/Read/ReadVariableOpReadVariableOpconv2d_84/kernel*&
_output_shapes
:*
dtype0
t
conv2d_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_84/bias
m
"conv2d_84/bias/Read/ReadVariableOpReadVariableOpconv2d_84/bias*
_output_shapes
:*
dtype0

conv2d_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_85/kernel
}
$conv2d_85/kernel/Read/ReadVariableOpReadVariableOpconv2d_85/kernel*&
_output_shapes
:*
dtype0
t
conv2d_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_85/bias
m
"conv2d_85/bias/Read/ReadVariableOpReadVariableOpconv2d_85/bias*
_output_shapes
:*
dtype0

conv2d_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_86/kernel
}
$conv2d_86/kernel/Read/ReadVariableOpReadVariableOpconv2d_86/kernel*&
_output_shapes
:*
dtype0
t
conv2d_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_86/bias
m
"conv2d_86/bias/Read/ReadVariableOpReadVariableOpconv2d_86/bias*
_output_shapes
:*
dtype0

conv2d_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_87/kernel
}
$conv2d_87/kernel/Read/ReadVariableOpReadVariableOpconv2d_87/kernel*&
_output_shapes
:*
dtype0
t
conv2d_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_87/bias
m
"conv2d_87/bias/Read/ReadVariableOpReadVariableOpconv2d_87/bias*
_output_shapes
:*
dtype0

conv2d_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_88/kernel
}
$conv2d_88/kernel/Read/ReadVariableOpReadVariableOpconv2d_88/kernel*&
_output_shapes
:*
dtype0
t
conv2d_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_88/bias
m
"conv2d_88/bias/Read/ReadVariableOpReadVariableOpconv2d_88/bias*
_output_shapes
:*
dtype0

conv2d_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_89/kernel
}
$conv2d_89/kernel/Read/ReadVariableOpReadVariableOpconv2d_89/kernel*&
_output_shapes
:*
dtype0
t
conv2d_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_89/bias
m
"conv2d_89/bias/Read/ReadVariableOpReadVariableOpconv2d_89/bias*
_output_shapes
:*
dtype0

conv2d_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_90/kernel
}
$conv2d_90/kernel/Read/ReadVariableOpReadVariableOpconv2d_90/kernel*&
_output_shapes
:*
dtype0
t
conv2d_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_90/bias
m
"conv2d_90/bias/Read/ReadVariableOpReadVariableOpconv2d_90/bias*
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

Adam/conv2d_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_84/kernel/m

+Adam/conv2d_84/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_84/bias/m
{
)Adam/conv2d_84/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_85/kernel/m

+Adam/conv2d_85/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_85/bias/m
{
)Adam/conv2d_85/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_86/kernel/m

+Adam/conv2d_86/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_86/bias/m
{
)Adam/conv2d_86/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_87/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_87/kernel/m

+Adam/conv2d_87/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_87/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_87/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_87/bias/m
{
)Adam/conv2d_87/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_87/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_88/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_88/kernel/m

+Adam/conv2d_88/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_88/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_88/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_88/bias/m
{
)Adam/conv2d_88/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_88/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_89/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_89/kernel/m

+Adam/conv2d_89/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_89/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_89/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_89/bias/m
{
)Adam/conv2d_89/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_89/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_90/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_90/kernel/m

+Adam/conv2d_90/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_90/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_90/bias/m
{
)Adam/conv2d_90/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_84/kernel/v

+Adam/conv2d_84/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_84/bias/v
{
)Adam/conv2d_84/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_84/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_85/kernel/v

+Adam/conv2d_85/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_85/bias/v
{
)Adam/conv2d_85/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_85/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_86/kernel/v

+Adam/conv2d_86/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_86/bias/v
{
)Adam/conv2d_86/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_86/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_87/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_87/kernel/v

+Adam/conv2d_87/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_87/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_87/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_87/bias/v
{
)Adam/conv2d_87/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_87/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_88/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_88/kernel/v

+Adam/conv2d_88/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_88/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_88/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_88/bias/v
{
)Adam/conv2d_88/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_88/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_89/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_89/kernel/v

+Adam/conv2d_89/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_89/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_89/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_89/bias/v
{
)Adam/conv2d_89/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_89/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_90/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_90/kernel/v

+Adam/conv2d_90/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_90/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_90/bias/v
{
)Adam/conv2d_90/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ós
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*®s
value¤sB¡s Bs
¤
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
¬
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
Ó
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
Ü
&iter

'beta_1

(beta_2
	)decay
*learning_rate+mã,mä-må.mæ/mç0mè1mé2mê3më4mì5mí6mî7mï8mð+vñ,vò-vó.vô/võ0vö1v÷2vø3vù4vú5vû6vü7vý8vþ*
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
°
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
¦

+kernel
,bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*

E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
¦

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*

Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses* 
¦

/kernel
0bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses*

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

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
¦

1kernel
2bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*

n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
¦

3kernel
4bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*

z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
¬

5kernel
6bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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
VARIABLE_VALUEconv2d_84/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_84/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_85/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_85/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_86/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_86/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_87/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_87/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_88/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_88/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_89/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_89/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv2d_90/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_90/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
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

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
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

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
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

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
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

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
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

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
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

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
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

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
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

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
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

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

70
81*

70
81*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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

Útotal

Ûcount
Ü	variables
Ý	keras_api*
M

Þtotal

ßcount
à
_fn_kwargs
á	variables
â	keras_api*
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
Ú0
Û1*

Ü	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Þ0
ß1*

á	variables*
sm
VARIABLE_VALUEAdam/conv2d_84/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_84/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_85/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_85/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_86/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_86/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_87/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_87/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_88/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_88/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_89/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_89/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_90/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_90/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_84/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_84/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_85/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_85/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_86/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_86/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_87/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_87/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_88/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_88/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_89/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_89/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv2d_90/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_90/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_imgPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿàà
Æ
StatefulPartitionedCallStatefulPartitionedCallserving_default_imgconv2d_84/kernelconv2d_84/biasconv2d_85/kernelconv2d_85/biasconv2d_86/kernelconv2d_86/biasconv2d_87/kernelconv2d_87/biasconv2d_88/kernelconv2d_88/biasconv2d_89/kernelconv2d_89/biasconv2d_90/kernelconv2d_90/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_6123328
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv2d_84/kernel/Read/ReadVariableOp"conv2d_84/bias/Read/ReadVariableOp$conv2d_85/kernel/Read/ReadVariableOp"conv2d_85/bias/Read/ReadVariableOp$conv2d_86/kernel/Read/ReadVariableOp"conv2d_86/bias/Read/ReadVariableOp$conv2d_87/kernel/Read/ReadVariableOp"conv2d_87/bias/Read/ReadVariableOp$conv2d_88/kernel/Read/ReadVariableOp"conv2d_88/bias/Read/ReadVariableOp$conv2d_89/kernel/Read/ReadVariableOp"conv2d_89/bias/Read/ReadVariableOp$conv2d_90/kernel/Read/ReadVariableOp"conv2d_90/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_84/kernel/m/Read/ReadVariableOp)Adam/conv2d_84/bias/m/Read/ReadVariableOp+Adam/conv2d_85/kernel/m/Read/ReadVariableOp)Adam/conv2d_85/bias/m/Read/ReadVariableOp+Adam/conv2d_86/kernel/m/Read/ReadVariableOp)Adam/conv2d_86/bias/m/Read/ReadVariableOp+Adam/conv2d_87/kernel/m/Read/ReadVariableOp)Adam/conv2d_87/bias/m/Read/ReadVariableOp+Adam/conv2d_88/kernel/m/Read/ReadVariableOp)Adam/conv2d_88/bias/m/Read/ReadVariableOp+Adam/conv2d_89/kernel/m/Read/ReadVariableOp)Adam/conv2d_89/bias/m/Read/ReadVariableOp+Adam/conv2d_90/kernel/m/Read/ReadVariableOp)Adam/conv2d_90/bias/m/Read/ReadVariableOp+Adam/conv2d_84/kernel/v/Read/ReadVariableOp)Adam/conv2d_84/bias/v/Read/ReadVariableOp+Adam/conv2d_85/kernel/v/Read/ReadVariableOp)Adam/conv2d_85/bias/v/Read/ReadVariableOp+Adam/conv2d_86/kernel/v/Read/ReadVariableOp)Adam/conv2d_86/bias/v/Read/ReadVariableOp+Adam/conv2d_87/kernel/v/Read/ReadVariableOp)Adam/conv2d_87/bias/v/Read/ReadVariableOp+Adam/conv2d_88/kernel/v/Read/ReadVariableOp)Adam/conv2d_88/bias/v/Read/ReadVariableOp+Adam/conv2d_89/kernel/v/Read/ReadVariableOp)Adam/conv2d_89/bias/v/Read/ReadVariableOp+Adam/conv2d_90/kernel/v/Read/ReadVariableOp)Adam/conv2d_90/bias/v/Read/ReadVariableOpConst*@
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_6123945
À

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_84/kernelconv2d_84/biasconv2d_85/kernelconv2d_85/biasconv2d_86/kernelconv2d_86/biasconv2d_87/kernelconv2d_87/biasconv2d_88/kernelconv2d_88/biasconv2d_89/kernelconv2d_89/biasconv2d_90/kernelconv2d_90/biastotalcounttotal_1count_1Adam/conv2d_84/kernel/mAdam/conv2d_84/bias/mAdam/conv2d_85/kernel/mAdam/conv2d_85/bias/mAdam/conv2d_86/kernel/mAdam/conv2d_86/bias/mAdam/conv2d_87/kernel/mAdam/conv2d_87/bias/mAdam/conv2d_88/kernel/mAdam/conv2d_88/bias/mAdam/conv2d_89/kernel/mAdam/conv2d_89/bias/mAdam/conv2d_90/kernel/mAdam/conv2d_90/bias/mAdam/conv2d_84/kernel/vAdam/conv2d_84/bias/vAdam/conv2d_85/kernel/vAdam/conv2d_85/bias/vAdam/conv2d_86/kernel/vAdam/conv2d_86/bias/vAdam/conv2d_87/kernel/vAdam/conv2d_87/bias/vAdam/conv2d_88/kernel/vAdam/conv2d_88/bias/vAdam/conv2d_89/kernel/vAdam/conv2d_89/bias/vAdam/conv2d_90/kernel/vAdam/conv2d_90/bias/v*?
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_6124108Ãé
"
º
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122815
conv2d_87_input+
conv2d_87_6122791:
conv2d_87_6122793:+
conv2d_88_6122797:
conv2d_88_6122799:+
conv2d_89_6122803:
conv2d_89_6122805:+
conv2d_90_6122809:
conv2d_90_6122811:
identity¢!conv2d_87/StatefulPartitionedCall¢!conv2d_88/StatefulPartitionedCall¢!conv2d_89/StatefulPartitionedCall¢!conv2d_90/StatefulPartitionedCall
!conv2d_87/StatefulPartitionedCallStatefulPartitionedCallconv2d_87_inputconv2d_87_6122791conv2d_87_6122793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551
 up_sampling2d_36/PartitionedCallPartitionedCall*conv2d_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492·
!conv2d_88/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_36/PartitionedCall:output:0conv2d_88_6122797conv2d_88_6122799*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569
 up_sampling2d_37/PartitionedCallPartitionedCall*conv2d_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511·
!conv2d_89/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_37/PartitionedCall:output:0conv2d_89_6122803conv2d_89_6122805*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587
 up_sampling2d_38/PartitionedCallPartitionedCall*conv2d_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530·
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_38/PartitionedCall:output:0conv2d_90_6122809conv2d_90_6122811*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605
IdentityIdentity*conv2d_90/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp"^conv2d_87/StatefulPartitionedCall"^conv2d_88/StatefulPartitionedCall"^conv2d_89/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2F
!conv2d_87/StatefulPartitionedCall!conv2d_87/StatefulPartitionedCall2F
!conv2d_88/StatefulPartitionedCall!conv2d_88/StatefulPartitionedCall2F
!conv2d_89/StatefulPartitionedCall!conv2d_89/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv2d_87_input
#
¯
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123390

inputsB
(conv2d_84_conv2d_readvariableop_resource:7
)conv2d_84_biasadd_readvariableop_resource:B
(conv2d_85_conv2d_readvariableop_resource:7
)conv2d_85_biasadd_readvariableop_resource:B
(conv2d_86_conv2d_readvariableop_resource:7
)conv2d_86_biasadd_readvariableop_resource:
identity¢ conv2d_84/BiasAdd/ReadVariableOp¢conv2d_84/Conv2D/ReadVariableOp¢ conv2d_85/BiasAdd/ReadVariableOp¢conv2d_85/Conv2D/ReadVariableOp¢ conv2d_86/BiasAdd/ReadVariableOp¢conv2d_86/Conv2D/ReadVariableOp
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
conv2d_84/Conv2DConv2Dinputs'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides

 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààt
conv2d_84/SigmoidSigmoidconv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà§
max_pooling2d_36/MaxPoolMaxPoolconv2d_84/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
ksize
*
paddingVALID*
strides

conv2d_85/Conv2D/ReadVariableOpReadVariableOp(conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0È
conv2d_85/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_85/BiasAdd/ReadVariableOpReadVariableOp)conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_85/BiasAddBiasAddconv2d_85/Conv2D:output:0(conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppl
conv2d_85/ReluReluconv2d_85/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp®
max_pooling2d_37/MaxPoolMaxPoolconv2d_85/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0È
conv2d_86/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88l
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88®
max_pooling2d_38/MaxPoolMaxPoolconv2d_86/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity!max_pooling2d_38/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp!^conv2d_85/BiasAdd/ReadVariableOp ^conv2d_85/Conv2D/ReadVariableOp!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2D
 conv2d_85/BiasAdd/ReadVariableOp conv2d_85/BiasAdd/ReadVariableOp2B
conv2d_85/Conv2D/ReadVariableOpconv2d_85/Conv2D/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6123608

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
ÿ
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6123769

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
ÿ
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
N
2__inference_max_pooling2d_38_layer_call_fn_6123633

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ	
 
/__inference_sequential_37_layer_call_fn_6123362

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122400w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ð	
©
/__inference_sequential_37_layer_call_fn_6122432
conv2d_84_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallconv2d_84_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122400w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
)
_user_specified_nameconv2d_84_input
ñ
ÿ
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

ç
/__inference_sequential_38_layer_call_fn_6122631
conv2d_87_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallconv2d_87_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122612
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv2d_87_input
ü
¿
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122314

inputs+
conv2d_84_6122271:
conv2d_84_6122273:+
conv2d_85_6122289:
conv2d_85_6122291:+
conv2d_86_6122307:
conv2d_86_6122309:
identity¢!conv2d_84/StatefulPartitionedCall¢!conv2d_85/StatefulPartitionedCall¢!conv2d_86/StatefulPartitionedCall
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_84_6122271conv2d_84_6122273*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270ø
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225¥
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_85_6122289conv2d_85_6122291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288ø
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237¥
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_86_6122307conv2d_86_6122309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306ø
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249
IdentityIdentity)max_pooling2d_38/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^conv2d_84/StatefulPartitionedCall"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ÿ!
±
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122612

inputs+
conv2d_87_6122552:
conv2d_87_6122554:+
conv2d_88_6122570:
conv2d_88_6122572:+
conv2d_89_6122588:
conv2d_89_6122590:+
conv2d_90_6122606:
conv2d_90_6122608:
identity¢!conv2d_87/StatefulPartitionedCall¢!conv2d_88/StatefulPartitionedCall¢!conv2d_89/StatefulPartitionedCall¢!conv2d_90/StatefulPartitionedCall
!conv2d_87/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_87_6122552conv2d_87_6122554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551
 up_sampling2d_36/PartitionedCallPartitionedCall*conv2d_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492·
!conv2d_88/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_36/PartitionedCall:output:0conv2d_88_6122570conv2d_88_6122572*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569
 up_sampling2d_37/PartitionedCallPartitionedCall*conv2d_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511·
!conv2d_89/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_37/PartitionedCall:output:0conv2d_89_6122588conv2d_89_6122590*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587
 up_sampling2d_38/PartitionedCallPartitionedCall*conv2d_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530·
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_38/PartitionedCall:output:0conv2d_90_6122606conv2d_90_6122608*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605
IdentityIdentity*conv2d_90/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp"^conv2d_87/StatefulPartitionedCall"^conv2d_88/StatefulPartitionedCall"^conv2d_89/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2F
!conv2d_87/StatefulPartitionedCall!conv2d_87/StatefulPartitionedCall2F
!conv2d_88/StatefulPartitionedCall!conv2d_88/StatefulPartitionedCall2F
!conv2d_89/StatefulPartitionedCall!conv2d_89/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6123568

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
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
:ÿÿÿÿÿÿÿÿÿàà`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ÖÇ
¥ 
#__inference__traced_restore_6124108
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: =
#assignvariableop_5_conv2d_84_kernel:/
!assignvariableop_6_conv2d_84_bias:=
#assignvariableop_7_conv2d_85_kernel:/
!assignvariableop_8_conv2d_85_bias:=
#assignvariableop_9_conv2d_86_kernel:0
"assignvariableop_10_conv2d_86_bias:>
$assignvariableop_11_conv2d_87_kernel:0
"assignvariableop_12_conv2d_87_bias:>
$assignvariableop_13_conv2d_88_kernel:0
"assignvariableop_14_conv2d_88_bias:>
$assignvariableop_15_conv2d_89_kernel:0
"assignvariableop_16_conv2d_89_bias:>
$assignvariableop_17_conv2d_90_kernel:0
"assignvariableop_18_conv2d_90_bias:#
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: E
+assignvariableop_23_adam_conv2d_84_kernel_m:7
)assignvariableop_24_adam_conv2d_84_bias_m:E
+assignvariableop_25_adam_conv2d_85_kernel_m:7
)assignvariableop_26_adam_conv2d_85_bias_m:E
+assignvariableop_27_adam_conv2d_86_kernel_m:7
)assignvariableop_28_adam_conv2d_86_bias_m:E
+assignvariableop_29_adam_conv2d_87_kernel_m:7
)assignvariableop_30_adam_conv2d_87_bias_m:E
+assignvariableop_31_adam_conv2d_88_kernel_m:7
)assignvariableop_32_adam_conv2d_88_bias_m:E
+assignvariableop_33_adam_conv2d_89_kernel_m:7
)assignvariableop_34_adam_conv2d_89_bias_m:E
+assignvariableop_35_adam_conv2d_90_kernel_m:7
)assignvariableop_36_adam_conv2d_90_bias_m:E
+assignvariableop_37_adam_conv2d_84_kernel_v:7
)assignvariableop_38_adam_conv2d_84_bias_v:E
+assignvariableop_39_adam_conv2d_85_kernel_v:7
)assignvariableop_40_adam_conv2d_85_bias_v:E
+assignvariableop_41_adam_conv2d_86_kernel_v:7
)assignvariableop_42_adam_conv2d_86_bias_v:E
+assignvariableop_43_adam_conv2d_87_kernel_v:7
)assignvariableop_44_adam_conv2d_87_bias_v:E
+assignvariableop_45_adam_conv2d_88_kernel_v:7
)assignvariableop_46_adam_conv2d_88_bias_v:E
+assignvariableop_47_adam_conv2d_89_kernel_v:7
)assignvariableop_48_adam_conv2d_89_bias_v:E
+assignvariableop_49_adam_conv2d_90_kernel_v:7
)assignvariableop_50_adam_conv2d_90_bias_v:
identity_52¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueB4B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*æ
_output_shapesÓ
Ð::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_84_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv2d_84_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_85_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_85_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_86_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_86_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_87_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_87_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_88_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_88_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_89_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_89_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv2d_90_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv2d_90_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_84_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_84_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_85_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_85_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_86_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_86_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_87_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_87_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_88_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_88_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_89_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_89_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_90_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_90_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_84_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_84_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_85_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_85_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_86_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_86_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_87_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_87_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_88_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_88_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_89_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_89_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_90_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_90_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ±	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 	
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
²

Þ
/__inference_sequential_38_layer_call_fn_6123439

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122612
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
 
+__inference_conv2d_85_layer_call_fn_6123587

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
°

J__inference_sequential_36_layer_call_and_return_conditional_losses_6123085
img/
sequential_37_6123054:#
sequential_37_6123056:/
sequential_37_6123058:#
sequential_37_6123060:/
sequential_37_6123062:#
sequential_37_6123064:/
sequential_38_6123067:#
sequential_38_6123069:/
sequential_38_6123071:#
sequential_38_6123073:/
sequential_38_6123075:#
sequential_38_6123077:/
sequential_38_6123079:#
sequential_38_6123081:
identity¢%sequential_37/StatefulPartitionedCall¢%sequential_38/StatefulPartitionedCalló
%sequential_37/StatefulPartitionedCallStatefulPartitionedCallimgsequential_37_6123054sequential_37_6123056sequential_37_6123058sequential_37_6123060sequential_37_6123062sequential_37_6123064*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122400â
%sequential_38/StatefulPartitionedCallStatefulPartitionedCall.sequential_37/StatefulPartitionedCall:output:0sequential_38_6123067sequential_38_6123069sequential_38_6123071sequential_38_6123073sequential_38_6123075sequential_38_6123077sequential_38_6123079sequential_38_6123081*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122721
IdentityIdentity.sequential_38/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^sequential_37/StatefulPartitionedCall&^sequential_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2N
%sequential_37/StatefulPartitionedCall%sequential_37/StatefulPartitionedCall2N
%sequential_38/StatefulPartitionedCall%sequential_38/StatefulPartitionedCall:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
²

Þ
/__inference_sequential_38_layer_call_fn_6123460

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122721
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

È
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122476
conv2d_84_input+
conv2d_84_6122457:
conv2d_84_6122459:+
conv2d_85_6122463:
conv2d_85_6122465:+
conv2d_86_6122469:
conv2d_86_6122471:
identity¢!conv2d_84/StatefulPartitionedCall¢!conv2d_85/StatefulPartitionedCall¢!conv2d_86/StatefulPartitionedCall
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCallconv2d_84_inputconv2d_84_6122457conv2d_84_6122459*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270ø
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225¥
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_85_6122463conv2d_85_6122465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288ø
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237¥
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_86_6122469conv2d_86_6122471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306ø
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249
IdentityIdentity)max_pooling2d_38/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^conv2d_84/StatefulPartitionedCall"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall:b ^
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
)
_user_specified_nameconv2d_84_input
¿
N
2__inference_up_sampling2d_36_layer_call_fn_6123663

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6123675

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µc

 __inference__traced_save_6123945
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv2d_84_kernel_read_readvariableop-
)savev2_conv2d_84_bias_read_readvariableop/
+savev2_conv2d_85_kernel_read_readvariableop-
)savev2_conv2d_85_bias_read_readvariableop/
+savev2_conv2d_86_kernel_read_readvariableop-
)savev2_conv2d_86_bias_read_readvariableop/
+savev2_conv2d_87_kernel_read_readvariableop-
)savev2_conv2d_87_bias_read_readvariableop/
+savev2_conv2d_88_kernel_read_readvariableop-
)savev2_conv2d_88_bias_read_readvariableop/
+savev2_conv2d_89_kernel_read_readvariableop-
)savev2_conv2d_89_bias_read_readvariableop/
+savev2_conv2d_90_kernel_read_readvariableop-
)savev2_conv2d_90_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_84_kernel_m_read_readvariableop4
0savev2_adam_conv2d_84_bias_m_read_readvariableop6
2savev2_adam_conv2d_85_kernel_m_read_readvariableop4
0savev2_adam_conv2d_85_bias_m_read_readvariableop6
2savev2_adam_conv2d_86_kernel_m_read_readvariableop4
0savev2_adam_conv2d_86_bias_m_read_readvariableop6
2savev2_adam_conv2d_87_kernel_m_read_readvariableop4
0savev2_adam_conv2d_87_bias_m_read_readvariableop6
2savev2_adam_conv2d_88_kernel_m_read_readvariableop4
0savev2_adam_conv2d_88_bias_m_read_readvariableop6
2savev2_adam_conv2d_89_kernel_m_read_readvariableop4
0savev2_adam_conv2d_89_bias_m_read_readvariableop6
2savev2_adam_conv2d_90_kernel_m_read_readvariableop4
0savev2_adam_conv2d_90_bias_m_read_readvariableop6
2savev2_adam_conv2d_84_kernel_v_read_readvariableop4
0savev2_adam_conv2d_84_bias_v_read_readvariableop6
2savev2_adam_conv2d_85_kernel_v_read_readvariableop4
0savev2_adam_conv2d_85_bias_v_read_readvariableop6
2savev2_adam_conv2d_86_kernel_v_read_readvariableop4
0savev2_adam_conv2d_86_bias_v_read_readvariableop6
2savev2_adam_conv2d_87_kernel_v_read_readvariableop4
0savev2_adam_conv2d_87_bias_v_read_readvariableop6
2savev2_adam_conv2d_88_kernel_v_read_readvariableop4
0savev2_adam_conv2d_88_bias_v_read_readvariableop6
2savev2_adam_conv2d_89_kernel_v_read_readvariableop4
0savev2_adam_conv2d_89_bias_v_read_readvariableop6
2savev2_adam_conv2d_90_kernel_v_read_readvariableop4
0savev2_adam_conv2d_90_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ó
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueB4B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ·
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv2d_84_kernel_read_readvariableop)savev2_conv2d_84_bias_read_readvariableop+savev2_conv2d_85_kernel_read_readvariableop)savev2_conv2d_85_bias_read_readvariableop+savev2_conv2d_86_kernel_read_readvariableop)savev2_conv2d_86_bias_read_readvariableop+savev2_conv2d_87_kernel_read_readvariableop)savev2_conv2d_87_bias_read_readvariableop+savev2_conv2d_88_kernel_read_readvariableop)savev2_conv2d_88_bias_read_readvariableop+savev2_conv2d_89_kernel_read_readvariableop)savev2_conv2d_89_bias_read_readvariableop+savev2_conv2d_90_kernel_read_readvariableop)savev2_conv2d_90_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_84_kernel_m_read_readvariableop0savev2_adam_conv2d_84_bias_m_read_readvariableop2savev2_adam_conv2d_85_kernel_m_read_readvariableop0savev2_adam_conv2d_85_bias_m_read_readvariableop2savev2_adam_conv2d_86_kernel_m_read_readvariableop0savev2_adam_conv2d_86_bias_m_read_readvariableop2savev2_adam_conv2d_87_kernel_m_read_readvariableop0savev2_adam_conv2d_87_bias_m_read_readvariableop2savev2_adam_conv2d_88_kernel_m_read_readvariableop0savev2_adam_conv2d_88_bias_m_read_readvariableop2savev2_adam_conv2d_89_kernel_m_read_readvariableop0savev2_adam_conv2d_89_bias_m_read_readvariableop2savev2_adam_conv2d_90_kernel_m_read_readvariableop0savev2_adam_conv2d_90_bias_m_read_readvariableop2savev2_adam_conv2d_84_kernel_v_read_readvariableop0savev2_adam_conv2d_84_bias_v_read_readvariableop2savev2_adam_conv2d_85_kernel_v_read_readvariableop0savev2_adam_conv2d_85_bias_v_read_readvariableop2savev2_adam_conv2d_86_kernel_v_read_readvariableop0savev2_adam_conv2d_86_bias_v_read_readvariableop2savev2_adam_conv2d_87_kernel_v_read_readvariableop0savev2_adam_conv2d_87_bias_v_read_readvariableop2savev2_adam_conv2d_88_kernel_v_read_readvariableop0savev2_adam_conv2d_88_bias_v_read_readvariableop2savev2_adam_conv2d_89_kernel_v_read_readvariableop0savev2_adam_conv2d_89_bias_v_read_readvariableop2savev2_adam_conv2d_90_kernel_v_read_readvariableop0savev2_adam_conv2d_90_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*£
_input_shapes
: : : : : : ::::::::::::::: : : : ::::::::::::::::::::::::::::: 2(
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
°

J__inference_sequential_36_layer_call_and_return_conditional_losses_6123051
img/
sequential_37_6123020:#
sequential_37_6123022:/
sequential_37_6123024:#
sequential_37_6123026:/
sequential_37_6123028:#
sequential_37_6123030:/
sequential_38_6123033:#
sequential_38_6123035:/
sequential_38_6123037:#
sequential_38_6123039:/
sequential_38_6123041:#
sequential_38_6123043:/
sequential_38_6123045:#
sequential_38_6123047:
identity¢%sequential_37/StatefulPartitionedCall¢%sequential_38/StatefulPartitionedCalló
%sequential_37/StatefulPartitionedCallStatefulPartitionedCallimgsequential_37_6123020sequential_37_6123022sequential_37_6123024sequential_37_6123026sequential_37_6123028sequential_37_6123030*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122314â
%sequential_38/StatefulPartitionedCallStatefulPartitionedCall.sequential_37/StatefulPartitionedCall:output:0sequential_38_6123033sequential_38_6123035sequential_38_6123037sequential_38_6123039sequential_38_6123041sequential_38_6123043sequential_38_6123045sequential_38_6123047*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122612
IdentityIdentity.sequential_38/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^sequential_37/StatefulPartitionedCall&^sequential_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2N
%sequential_37/StatefulPartitionedCall%sequential_37/StatefulPartitionedCall2N
%sequential_38/StatefulPartitionedCall%sequential_38/StatefulPartitionedCall:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
"
º
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122788
conv2d_87_input+
conv2d_87_6122764:
conv2d_87_6122766:+
conv2d_88_6122770:
conv2d_88_6122772:+
conv2d_89_6122776:
conv2d_89_6122778:+
conv2d_90_6122782:
conv2d_90_6122784:
identity¢!conv2d_87/StatefulPartitionedCall¢!conv2d_88/StatefulPartitionedCall¢!conv2d_89/StatefulPartitionedCall¢!conv2d_90/StatefulPartitionedCall
!conv2d_87/StatefulPartitionedCallStatefulPartitionedCallconv2d_87_inputconv2d_87_6122764conv2d_87_6122766*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551
 up_sampling2d_36/PartitionedCallPartitionedCall*conv2d_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492·
!conv2d_88/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_36/PartitionedCall:output:0conv2d_88_6122770conv2d_88_6122772*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569
 up_sampling2d_37/PartitionedCallPartitionedCall*conv2d_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511·
!conv2d_89/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_37/PartitionedCall:output:0conv2d_89_6122776conv2d_89_6122778*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587
 up_sampling2d_38/PartitionedCallPartitionedCall*conv2d_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530·
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_38/PartitionedCall:output:0conv2d_90_6122782conv2d_90_6122784*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605
IdentityIdentity*conv2d_90/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp"^conv2d_87/StatefulPartitionedCall"^conv2d_88/StatefulPartitionedCall"^conv2d_89/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2F
!conv2d_87/StatefulPartitionedCall!conv2d_87/StatefulPartitionedCall2F
!conv2d_88/StatefulPartitionedCall!conv2d_88/StatefulPartitionedCall2F
!conv2d_89/StatefulPartitionedCall!conv2d_89/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv2d_87_input
ò
ÿ
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ5
ñ
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123504

inputsB
(conv2d_87_conv2d_readvariableop_resource:7
)conv2d_87_biasadd_readvariableop_resource:B
(conv2d_88_conv2d_readvariableop_resource:7
)conv2d_88_biasadd_readvariableop_resource:B
(conv2d_89_conv2d_readvariableop_resource:7
)conv2d_89_biasadd_readvariableop_resource:B
(conv2d_90_conv2d_readvariableop_resource:7
)conv2d_90_biasadd_readvariableop_resource:
identity¢ conv2d_87/BiasAdd/ReadVariableOp¢conv2d_87/Conv2D/ReadVariableOp¢ conv2d_88/BiasAdd/ReadVariableOp¢conv2d_88/Conv2D/ReadVariableOp¢ conv2d_89/BiasAdd/ReadVariableOp¢conv2d_89/Conv2D/ReadVariableOp¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_87/Conv2DConv2Dinputs'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
up_sampling2d_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_36/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_36/mulMulup_sampling2d_36/Const:output:0!up_sampling2d_36/Const_1:output:0*
T0*
_output_shapes
:Ò
-up_sampling2d_36/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_87/Relu:activations:0up_sampling2d_36/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
half_pixel_centers(
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
conv2d_88/Conv2DConv2D>up_sampling2d_36/resize/ResizeNearestNeighbor:resized_images:0'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88l
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88g
up_sampling2d_37/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   i
up_sampling2d_37/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_37/mulMulup_sampling2d_37/Const:output:0!up_sampling2d_37/Const_1:output:0*
T0*
_output_shapes
:Ò
-up_sampling2d_37/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_88/Relu:activations:0up_sampling2d_37/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
half_pixel_centers(
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
conv2d_89/Conv2DConv2D>up_sampling2d_37/resize/ResizeNearestNeighbor:resized_images:0'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppl
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppg
up_sampling2d_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   i
up_sampling2d_38/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_38/mulMulup_sampling2d_38/Const:output:0!up_sampling2d_38/Const_1:output:0*
T0*
_output_shapes
:Ô
-up_sampling2d_38/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_89/Relu:activations:0up_sampling2d_38/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ç
conv2d_90/Conv2DConv2D>up_sampling2d_38/resize/ResizeNearestNeighbor:resized_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides

 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààt
conv2d_90/SigmoidSigmoidconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààn
IdentityIdentityconv2d_90/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÚ
NoOpNoOp!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
 
+__inference_conv2d_84_layer_call_fn_6123557

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
¹

J__inference_sequential_36_layer_call_and_return_conditional_losses_6122853

inputs/
sequential_37_6122822:#
sequential_37_6122824:/
sequential_37_6122826:#
sequential_37_6122828:/
sequential_37_6122830:#
sequential_37_6122832:/
sequential_38_6122835:#
sequential_38_6122837:/
sequential_38_6122839:#
sequential_38_6122841:/
sequential_38_6122843:#
sequential_38_6122845:/
sequential_38_6122847:#
sequential_38_6122849:
identity¢%sequential_37/StatefulPartitionedCall¢%sequential_38/StatefulPartitionedCallö
%sequential_37/StatefulPartitionedCallStatefulPartitionedCallinputssequential_37_6122822sequential_37_6122824sequential_37_6122826sequential_37_6122828sequential_37_6122830sequential_37_6122832*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122314â
%sequential_38/StatefulPartitionedCallStatefulPartitionedCall.sequential_37/StatefulPartitionedCall:output:0sequential_38_6122835sequential_38_6122837sequential_38_6122839sequential_38_6122841sequential_38_6122843sequential_38_6122845sequential_38_6122847sequential_38_6122849*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122612
IdentityIdentity.sequential_38/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^sequential_37/StatefulPartitionedCall&^sequential_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2N
%sequential_37/StatefulPartitionedCall%sequential_37/StatefulPartitionedCall2N
%sequential_38/StatefulPartitionedCall%sequential_38/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ç

/__inference_sequential_36_layer_call_fn_6122884
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
identity¢StatefulPartitionedCall
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_36_layer_call_and_return_conditional_losses_6122853
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
¼i
¿
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123225

inputsP
6sequential_37_conv2d_84_conv2d_readvariableop_resource:E
7sequential_37_conv2d_84_biasadd_readvariableop_resource:P
6sequential_37_conv2d_85_conv2d_readvariableop_resource:E
7sequential_37_conv2d_85_biasadd_readvariableop_resource:P
6sequential_37_conv2d_86_conv2d_readvariableop_resource:E
7sequential_37_conv2d_86_biasadd_readvariableop_resource:P
6sequential_38_conv2d_87_conv2d_readvariableop_resource:E
7sequential_38_conv2d_87_biasadd_readvariableop_resource:P
6sequential_38_conv2d_88_conv2d_readvariableop_resource:E
7sequential_38_conv2d_88_biasadd_readvariableop_resource:P
6sequential_38_conv2d_89_conv2d_readvariableop_resource:E
7sequential_38_conv2d_89_biasadd_readvariableop_resource:P
6sequential_38_conv2d_90_conv2d_readvariableop_resource:E
7sequential_38_conv2d_90_biasadd_readvariableop_resource:
identity¢.sequential_37/conv2d_84/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_84/Conv2D/ReadVariableOp¢.sequential_37/conv2d_85/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_85/Conv2D/ReadVariableOp¢.sequential_37/conv2d_86/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_86/Conv2D/ReadVariableOp¢.sequential_38/conv2d_87/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_87/Conv2D/ReadVariableOp¢.sequential_38/conv2d_88/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_88/Conv2D/ReadVariableOp¢.sequential_38/conv2d_89/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_89/Conv2D/ReadVariableOp¢.sequential_38/conv2d_90/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_90/Conv2D/ReadVariableOp¬
-sequential_37/conv2d_84/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ë
sequential_37/conv2d_84/Conv2DConv2Dinputs5sequential_37/conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¢
.sequential_37/conv2d_84/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
sequential_37/conv2d_84/BiasAddBiasAdd'sequential_37/conv2d_84/Conv2D:output:06sequential_37/conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
sequential_37/conv2d_84/SigmoidSigmoid(sequential_37/conv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÃ
&sequential_37/max_pooling2d_36/MaxPoolMaxPool#sequential_37/conv2d_84/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
ksize
*
paddingVALID*
strides
¬
-sequential_37/conv2d_85/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_37/conv2d_85/Conv2DConv2D/sequential_37/max_pooling2d_36/MaxPool:output:05sequential_37/conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¢
.sequential_37/conv2d_85/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_37/conv2d_85/BiasAddBiasAdd'sequential_37/conv2d_85/Conv2D:output:06sequential_37/conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
sequential_37/conv2d_85/ReluRelu(sequential_37/conv2d_85/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÊ
&sequential_37/max_pooling2d_37/MaxPoolMaxPool*sequential_37/conv2d_85/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¬
-sequential_37/conv2d_86/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_37/conv2d_86/Conv2DConv2D/sequential_37/max_pooling2d_37/MaxPool:output:05sequential_37/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¢
.sequential_37/conv2d_86/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_37/conv2d_86/BiasAddBiasAdd'sequential_37/conv2d_86/Conv2D:output:06sequential_37/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
sequential_37/conv2d_86/ReluRelu(sequential_37/conv2d_86/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ê
&sequential_37/max_pooling2d_38/MaxPoolMaxPool*sequential_37/conv2d_86/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¬
-sequential_38/conv2d_87/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_87_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_38/conv2d_87/Conv2DConv2D/sequential_37/max_pooling2d_38/MaxPool:output:05sequential_38/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¢
.sequential_38/conv2d_87/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_87/BiasAddBiasAdd'sequential_38/conv2d_87/Conv2D:output:06sequential_38/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_38/conv2d_87/ReluRelu(sequential_38/conv2d_87/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$sequential_38/up_sampling2d_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"      w
&sequential_38/up_sampling2d_36/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_36/mulMul-sequential_38/up_sampling2d_36/Const:output:0/sequential_38/up_sampling2d_36/Const_1:output:0*
T0*
_output_shapes
:ü
;sequential_38/up_sampling2d_36/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_87/Relu:activations:0&sequential_38/up_sampling2d_36/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
half_pixel_centers(¬
-sequential_38/conv2d_88/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_88_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_88/Conv2DConv2DLsequential_38/up_sampling2d_36/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¢
.sequential_38/conv2d_88/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_88/BiasAddBiasAdd'sequential_38/conv2d_88/Conv2D:output:06sequential_38/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
sequential_38/conv2d_88/ReluRelu(sequential_38/conv2d_88/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88u
$sequential_38/up_sampling2d_37/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   w
&sequential_38/up_sampling2d_37/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_37/mulMul-sequential_38/up_sampling2d_37/Const:output:0/sequential_38/up_sampling2d_37/Const_1:output:0*
T0*
_output_shapes
:ü
;sequential_38/up_sampling2d_37/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_88/Relu:activations:0&sequential_38/up_sampling2d_37/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
half_pixel_centers(¬
-sequential_38/conv2d_89/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_89_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_89/Conv2DConv2DLsequential_38/up_sampling2d_37/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¢
.sequential_38/conv2d_89/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_89/BiasAddBiasAdd'sequential_38/conv2d_89/Conv2D:output:06sequential_38/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
sequential_38/conv2d_89/ReluRelu(sequential_38/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppu
$sequential_38/up_sampling2d_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   w
&sequential_38/up_sampling2d_38/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_38/mulMul-sequential_38/up_sampling2d_38/Const:output:0/sequential_38/up_sampling2d_38/Const_1:output:0*
T0*
_output_shapes
:þ
;sequential_38/up_sampling2d_38/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_89/Relu:activations:0&sequential_38/up_sampling2d_38/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(¬
-sequential_38/conv2d_90/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_90/Conv2DConv2DLsequential_38/up_sampling2d_38/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¢
.sequential_38/conv2d_90/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
sequential_38/conv2d_90/BiasAddBiasAdd'sequential_38/conv2d_90/Conv2D:output:06sequential_38/conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
sequential_38/conv2d_90/SigmoidSigmoid(sequential_38/conv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà|
IdentityIdentity#sequential_38/conv2d_90/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààí
NoOpNoOp/^sequential_37/conv2d_84/BiasAdd/ReadVariableOp.^sequential_37/conv2d_84/Conv2D/ReadVariableOp/^sequential_37/conv2d_85/BiasAdd/ReadVariableOp.^sequential_37/conv2d_85/Conv2D/ReadVariableOp/^sequential_37/conv2d_86/BiasAdd/ReadVariableOp.^sequential_37/conv2d_86/Conv2D/ReadVariableOp/^sequential_38/conv2d_87/BiasAdd/ReadVariableOp.^sequential_38/conv2d_87/Conv2D/ReadVariableOp/^sequential_38/conv2d_88/BiasAdd/ReadVariableOp.^sequential_38/conv2d_88/Conv2D/ReadVariableOp/^sequential_38/conv2d_89/BiasAdd/ReadVariableOp.^sequential_38/conv2d_89/Conv2D/ReadVariableOp/^sequential_38/conv2d_90/BiasAdd/ReadVariableOp.^sequential_38/conv2d_90/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2`
.sequential_37/conv2d_84/BiasAdd/ReadVariableOp.sequential_37/conv2d_84/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_84/Conv2D/ReadVariableOp-sequential_37/conv2d_84/Conv2D/ReadVariableOp2`
.sequential_37/conv2d_85/BiasAdd/ReadVariableOp.sequential_37/conv2d_85/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_85/Conv2D/ReadVariableOp-sequential_37/conv2d_85/Conv2D/ReadVariableOp2`
.sequential_37/conv2d_86/BiasAdd/ReadVariableOp.sequential_37/conv2d_86/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_86/Conv2D/ReadVariableOp-sequential_37/conv2d_86/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_87/BiasAdd/ReadVariableOp.sequential_38/conv2d_87/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_87/Conv2D/ReadVariableOp-sequential_38/conv2d_87/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_88/BiasAdd/ReadVariableOp.sequential_38/conv2d_88/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_88/Conv2D/ReadVariableOp-sequential_38/conv2d_88/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_89/BiasAdd/ReadVariableOp.sequential_38/conv2d_89/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_89/Conv2D/ReadVariableOp-sequential_38/conv2d_89/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_90/BiasAdd/ReadVariableOp.sequential_38/conv2d_90/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_90/Conv2D/ReadVariableOp-sequential_38/conv2d_90/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6123598

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
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
:ÿÿÿÿÿÿÿÿÿppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
º
 
+__inference_conv2d_88_layer_call_fn_6123684

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6123658

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6123628

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
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
:ÿÿÿÿÿÿÿÿÿ88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

È
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122454
conv2d_84_input+
conv2d_84_6122435:
conv2d_84_6122437:+
conv2d_85_6122441:
conv2d_85_6122443:+
conv2d_86_6122447:
conv2d_86_6122449:
identity¢!conv2d_84/StatefulPartitionedCall¢!conv2d_85/StatefulPartitionedCall¢!conv2d_86/StatefulPartitionedCall
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCallconv2d_84_inputconv2d_84_6122435conv2d_84_6122437*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270ø
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225¥
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_85_6122441conv2d_85_6122443*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288ø
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237¥
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_86_6122447conv2d_86_6122449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306ø
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249
IdentityIdentity)max_pooling2d_38/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^conv2d_84/StatefulPartitionedCall"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall:b ^
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
)
_user_specified_nameconv2d_84_input

i
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6123638

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð	
©
/__inference_sequential_37_layer_call_fn_6122329
conv2d_84_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallconv2d_84_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122314w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
)
_user_specified_nameconv2d_84_input
º
 
+__inference_conv2d_89_layer_call_fn_6123721

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
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
:ÿÿÿÿÿÿÿÿÿàà`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

ç
/__inference_sequential_38_layer_call_fn_6122761
conv2d_87_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallconv2d_87_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122721
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameconv2d_87_input

i
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6123712

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
 
+__inference_conv2d_90_layer_call_fn_6123758

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

/__inference_sequential_36_layer_call_fn_6123157

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
identity¢StatefulPartitionedCall
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_36_layer_call_and_return_conditional_losses_6122953
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
µ	
 
/__inference_sequential_37_layer_call_fn_6123345

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122314w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ÿ!
±
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122721

inputs+
conv2d_87_6122697:
conv2d_87_6122699:+
conv2d_88_6122703:
conv2d_88_6122705:+
conv2d_89_6122709:
conv2d_89_6122711:+
conv2d_90_6122715:
conv2d_90_6122717:
identity¢!conv2d_87/StatefulPartitionedCall¢!conv2d_88/StatefulPartitionedCall¢!conv2d_89/StatefulPartitionedCall¢!conv2d_90/StatefulPartitionedCall
!conv2d_87/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_87_6122697conv2d_87_6122699*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551
 up_sampling2d_36/PartitionedCallPartitionedCall*conv2d_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6122492·
!conv2d_88/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_36/PartitionedCall:output:0conv2d_88_6122703conv2d_88_6122705*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6122569
 up_sampling2d_37/PartitionedCallPartitionedCall*conv2d_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511·
!conv2d_89/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_37/PartitionedCall:output:0conv2d_89_6122709conv2d_89_6122711*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6122587
 up_sampling2d_38/PartitionedCallPartitionedCall*conv2d_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530·
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_38/PartitionedCall:output:0conv2d_90_6122715conv2d_90_6122717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6122605
IdentityIdentity*conv2d_90/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
NoOpNoOp"^conv2d_87/StatefulPartitionedCall"^conv2d_88/StatefulPartitionedCall"^conv2d_89/StatefulPartitionedCall"^conv2d_90/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2F
!conv2d_87/StatefulPartitionedCall!conv2d_87/StatefulPartitionedCall2F
!conv2d_88/StatefulPartitionedCall!conv2d_88/StatefulPartitionedCall2F
!conv2d_89/StatefulPartitionedCall!conv2d_89/StatefulPartitionedCall2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6123749

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢}

"__inference__wrapped_model_6122216
img^
Dsequential_36_sequential_37_conv2d_84_conv2d_readvariableop_resource:S
Esequential_36_sequential_37_conv2d_84_biasadd_readvariableop_resource:^
Dsequential_36_sequential_37_conv2d_85_conv2d_readvariableop_resource:S
Esequential_36_sequential_37_conv2d_85_biasadd_readvariableop_resource:^
Dsequential_36_sequential_37_conv2d_86_conv2d_readvariableop_resource:S
Esequential_36_sequential_37_conv2d_86_biasadd_readvariableop_resource:^
Dsequential_36_sequential_38_conv2d_87_conv2d_readvariableop_resource:S
Esequential_36_sequential_38_conv2d_87_biasadd_readvariableop_resource:^
Dsequential_36_sequential_38_conv2d_88_conv2d_readvariableop_resource:S
Esequential_36_sequential_38_conv2d_88_biasadd_readvariableop_resource:^
Dsequential_36_sequential_38_conv2d_89_conv2d_readvariableop_resource:S
Esequential_36_sequential_38_conv2d_89_biasadd_readvariableop_resource:^
Dsequential_36_sequential_38_conv2d_90_conv2d_readvariableop_resource:S
Esequential_36_sequential_38_conv2d_90_biasadd_readvariableop_resource:
identity¢<sequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOp¢;sequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOp¢<sequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOp¢;sequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOp¢<sequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOp¢;sequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOp¢<sequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOp¢;sequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOp¢<sequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOp¢;sequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOp¢<sequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOp¢;sequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOp¢<sequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOp¢;sequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOpÈ
;sequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_37_conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ä
,sequential_36/sequential_37/conv2d_84/Conv2DConv2DimgCsequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¾
<sequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_37_conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ñ
-sequential_36/sequential_37/conv2d_84/BiasAddBiasAdd5sequential_36/sequential_37/conv2d_84/Conv2D:output:0Dsequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà¬
-sequential_36/sequential_37/conv2d_84/SigmoidSigmoid6sequential_36/sequential_37/conv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààß
4sequential_36/sequential_37/max_pooling2d_36/MaxPoolMaxPool1sequential_36/sequential_37/conv2d_84/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
ksize
*
paddingVALID*
strides
È
;sequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_37_conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
,sequential_36/sequential_37/conv2d_85/Conv2DConv2D=sequential_36/sequential_37/max_pooling2d_36/MaxPool:output:0Csequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¾
<sequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_37_conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ï
-sequential_36/sequential_37/conv2d_85/BiasAddBiasAdd5sequential_36/sequential_37/conv2d_85/Conv2D:output:0Dsequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
*sequential_36/sequential_37/conv2d_85/ReluRelu6sequential_36/sequential_37/conv2d_85/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppæ
4sequential_36/sequential_37/max_pooling2d_37/MaxPoolMaxPool8sequential_36/sequential_37/conv2d_85/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
È
;sequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_37_conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
,sequential_36/sequential_37/conv2d_86/Conv2DConv2D=sequential_36/sequential_37/max_pooling2d_37/MaxPool:output:0Csequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¾
<sequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_37_conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ï
-sequential_36/sequential_37/conv2d_86/BiasAddBiasAdd5sequential_36/sequential_37/conv2d_86/Conv2D:output:0Dsequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
*sequential_36/sequential_37/conv2d_86/ReluRelu6sequential_36/sequential_37/conv2d_86/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88æ
4sequential_36/sequential_37/max_pooling2d_38/MaxPoolMaxPool8sequential_36/sequential_37/conv2d_86/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
È
;sequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_38_conv2d_87_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
,sequential_36/sequential_38/conv2d_87/Conv2DConv2D=sequential_36/sequential_37/max_pooling2d_38/MaxPool:output:0Csequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¾
<sequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_38_conv2d_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ï
-sequential_36/sequential_38/conv2d_87/BiasAddBiasAdd5sequential_36/sequential_38/conv2d_87/Conv2D:output:0Dsequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
*sequential_36/sequential_38/conv2d_87/ReluRelu6sequential_36/sequential_38/conv2d_87/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2sequential_36/sequential_38/up_sampling2d_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"      
4sequential_36/sequential_38/up_sampling2d_36/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ø
0sequential_36/sequential_38/up_sampling2d_36/mulMul;sequential_36/sequential_38/up_sampling2d_36/Const:output:0=sequential_36/sequential_38/up_sampling2d_36/Const_1:output:0*
T0*
_output_shapes
:¦
Isequential_36/sequential_38/up_sampling2d_36/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_36/sequential_38/conv2d_87/Relu:activations:04sequential_36/sequential_38/up_sampling2d_36/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
half_pixel_centers(È
;sequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_38_conv2d_88_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
,sequential_36/sequential_38/conv2d_88/Conv2DConv2DZsequential_36/sequential_38/up_sampling2d_36/resize/ResizeNearestNeighbor:resized_images:0Csequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¾
<sequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_38_conv2d_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ï
-sequential_36/sequential_38/conv2d_88/BiasAddBiasAdd5sequential_36/sequential_38/conv2d_88/Conv2D:output:0Dsequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88¤
*sequential_36/sequential_38/conv2d_88/ReluRelu6sequential_36/sequential_38/conv2d_88/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
2sequential_36/sequential_38/up_sampling2d_37/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   
4sequential_36/sequential_38/up_sampling2d_37/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ø
0sequential_36/sequential_38/up_sampling2d_37/mulMul;sequential_36/sequential_38/up_sampling2d_37/Const:output:0=sequential_36/sequential_38/up_sampling2d_37/Const_1:output:0*
T0*
_output_shapes
:¦
Isequential_36/sequential_38/up_sampling2d_37/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_36/sequential_38/conv2d_88/Relu:activations:04sequential_36/sequential_38/up_sampling2d_37/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
half_pixel_centers(È
;sequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_38_conv2d_89_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¹
,sequential_36/sequential_38/conv2d_89/Conv2DConv2DZsequential_36/sequential_38/up_sampling2d_37/resize/ResizeNearestNeighbor:resized_images:0Csequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¾
<sequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_38_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ï
-sequential_36/sequential_38/conv2d_89/BiasAddBiasAdd5sequential_36/sequential_38/conv2d_89/Conv2D:output:0Dsequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp¤
*sequential_36/sequential_38/conv2d_89/ReluRelu6sequential_36/sequential_38/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
2sequential_36/sequential_38/up_sampling2d_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   
4sequential_36/sequential_38/up_sampling2d_38/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ø
0sequential_36/sequential_38/up_sampling2d_38/mulMul;sequential_36/sequential_38/up_sampling2d_38/Const:output:0=sequential_36/sequential_38/up_sampling2d_38/Const_1:output:0*
T0*
_output_shapes
:¨
Isequential_36/sequential_38/up_sampling2d_38/resize/ResizeNearestNeighborResizeNearestNeighbor8sequential_36/sequential_38/conv2d_89/Relu:activations:04sequential_36/sequential_38/up_sampling2d_38/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(È
;sequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOpReadVariableOpDsequential_36_sequential_38_conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0»
,sequential_36/sequential_38/conv2d_90/Conv2DConv2DZsequential_36/sequential_38/up_sampling2d_38/resize/ResizeNearestNeighbor:resized_images:0Csequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¾
<sequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOpReadVariableOpEsequential_36_sequential_38_conv2d_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ñ
-sequential_36/sequential_38/conv2d_90/BiasAddBiasAdd5sequential_36/sequential_38/conv2d_90/Conv2D:output:0Dsequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà¬
-sequential_36/sequential_38/conv2d_90/SigmoidSigmoid6sequential_36/sequential_38/conv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
IdentityIdentity1sequential_36/sequential_38/conv2d_90/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà±
NoOpNoOp=^sequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOp<^sequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOp=^sequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOp<^sequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOp=^sequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOp<^sequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOp=^sequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOp<^sequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOp=^sequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOp<^sequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOp=^sequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOp<^sequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOp=^sequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOp<^sequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2|
<sequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOp<sequential_36/sequential_37/conv2d_84/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOp;sequential_36/sequential_37/conv2d_84/Conv2D/ReadVariableOp2|
<sequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOp<sequential_36/sequential_37/conv2d_85/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOp;sequential_36/sequential_37/conv2d_85/Conv2D/ReadVariableOp2|
<sequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOp<sequential_36/sequential_37/conv2d_86/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOp;sequential_36/sequential_37/conv2d_86/Conv2D/ReadVariableOp2|
<sequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOp<sequential_36/sequential_38/conv2d_87/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOp;sequential_36/sequential_38/conv2d_87/Conv2D/ReadVariableOp2|
<sequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOp<sequential_36/sequential_38/conv2d_88/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOp;sequential_36/sequential_38/conv2d_88/Conv2D/ReadVariableOp2|
<sequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOp<sequential_36/sequential_38/conv2d_89/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOp;sequential_36/sequential_38/conv2d_89/Conv2D/ReadVariableOp2|
<sequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOp<sequential_36/sequential_38/conv2d_90/BiasAdd/ReadVariableOp2z
;sequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOp;sequential_36/sequential_38/conv2d_90/Conv2D/ReadVariableOp:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
#
¯
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123418

inputsB
(conv2d_84_conv2d_readvariableop_resource:7
)conv2d_84_biasadd_readvariableop_resource:B
(conv2d_85_conv2d_readvariableop_resource:7
)conv2d_85_biasadd_readvariableop_resource:B
(conv2d_86_conv2d_readvariableop_resource:7
)conv2d_86_biasadd_readvariableop_resource:
identity¢ conv2d_84/BiasAdd/ReadVariableOp¢conv2d_84/Conv2D/ReadVariableOp¢ conv2d_85/BiasAdd/ReadVariableOp¢conv2d_85/Conv2D/ReadVariableOp¢ conv2d_86/BiasAdd/ReadVariableOp¢conv2d_86/Conv2D/ReadVariableOp
conv2d_84/Conv2D/ReadVariableOpReadVariableOp(conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
conv2d_84/Conv2DConv2Dinputs'conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides

 conv2d_84/BiasAdd/ReadVariableOpReadVariableOp)conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_84/BiasAddBiasAddconv2d_84/Conv2D:output:0(conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààt
conv2d_84/SigmoidSigmoidconv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà§
max_pooling2d_36/MaxPoolMaxPoolconv2d_84/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
ksize
*
paddingVALID*
strides

conv2d_85/Conv2D/ReadVariableOpReadVariableOp(conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0È
conv2d_85/Conv2DConv2D!max_pooling2d_36/MaxPool:output:0'conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_85/BiasAdd/ReadVariableOpReadVariableOp)conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_85/BiasAddBiasAddconv2d_85/Conv2D:output:0(conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppl
conv2d_85/ReluReluconv2d_85/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp®
max_pooling2d_37/MaxPoolMaxPoolconv2d_85/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides

conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0È
conv2d_86/Conv2DConv2D!max_pooling2d_37/MaxPool:output:0'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88l
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88®
max_pooling2d_38/MaxPoolMaxPoolconv2d_86/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
x
IdentityIdentity!max_pooling2d_38/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^conv2d_84/BiasAdd/ReadVariableOp ^conv2d_84/Conv2D/ReadVariableOp!^conv2d_85/BiasAdd/ReadVariableOp ^conv2d_85/Conv2D/ReadVariableOp!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2D
 conv2d_84/BiasAdd/ReadVariableOp conv2d_84/BiasAdd/ReadVariableOp2B
conv2d_84/Conv2D/ReadVariableOpconv2d_84/Conv2D/ReadVariableOp2D
 conv2d_85/BiasAdd/ReadVariableOp conv2d_85/BiasAdd/ReadVariableOp2B
conv2d_85/Conv2D/ReadVariableOpconv2d_85/Conv2D/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
¹

J__inference_sequential_36_layer_call_and_return_conditional_losses_6122953

inputs/
sequential_37_6122922:#
sequential_37_6122924:/
sequential_37_6122926:#
sequential_37_6122928:/
sequential_37_6122930:#
sequential_37_6122932:/
sequential_38_6122935:#
sequential_38_6122937:/
sequential_38_6122939:#
sequential_38_6122941:/
sequential_38_6122943:#
sequential_38_6122945:/
sequential_38_6122947:#
sequential_38_6122949:
identity¢%sequential_37/StatefulPartitionedCall¢%sequential_38/StatefulPartitionedCallö
%sequential_37/StatefulPartitionedCallStatefulPartitionedCallinputssequential_37_6122922sequential_37_6122924sequential_37_6122926sequential_37_6122928sequential_37_6122930sequential_37_6122932*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122400â
%sequential_38/StatefulPartitionedCallStatefulPartitionedCall.sequential_37/StatefulPartitionedCall:output:0sequential_38_6122935sequential_38_6122937sequential_38_6122939sequential_38_6122941sequential_38_6122943sequential_38_6122945sequential_38_6122947sequential_38_6122949*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122721
IdentityIdentity.sequential_38/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp&^sequential_37/StatefulPartitionedCall&^sequential_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2N
%sequential_37/StatefulPartitionedCall%sequential_37/StatefulPartitionedCall2N
%sequential_38/StatefulPartitionedCall%sequential_38/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
¿
N
2__inference_max_pooling2d_36_layer_call_fn_6123573

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
¿
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122400

inputs+
conv2d_84_6122381:
conv2d_84_6122383:+
conv2d_85_6122387:
conv2d_85_6122389:+
conv2d_86_6122393:
conv2d_86_6122395:
identity¢!conv2d_84/StatefulPartitionedCall¢!conv2d_85/StatefulPartitionedCall¢!conv2d_86/StatefulPartitionedCall
!conv2d_84/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_84_6122381conv2d_84_6122383*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6122270ø
 max_pooling2d_36/PartitionedCallPartitionedCall*conv2d_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225¥
!conv2d_85/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_36/PartitionedCall:output:0conv2d_85_6122387conv2d_85_6122389*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288ø
 max_pooling2d_37/PartitionedCallPartitionedCall*conv2d_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237¥
!conv2d_86/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_37/PartitionedCall:output:0conv2d_86_6122393conv2d_86_6122395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306ø
 max_pooling2d_38/PartitionedCallPartitionedCall*conv2d_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249
IdentityIdentity)max_pooling2d_38/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
NoOpNoOp"^conv2d_84/StatefulPartitionedCall"^conv2d_85/StatefulPartitionedCall"^conv2d_86/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿàà: : : : : : 2F
!conv2d_84/StatefulPartitionedCall!conv2d_84/StatefulPartitionedCall2F
!conv2d_85/StatefulPartitionedCall!conv2d_85/StatefulPartitionedCall2F
!conv2d_86/StatefulPartitionedCall!conv2d_86/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
ñ5
ñ
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123548

inputsB
(conv2d_87_conv2d_readvariableop_resource:7
)conv2d_87_biasadd_readvariableop_resource:B
(conv2d_88_conv2d_readvariableop_resource:7
)conv2d_88_biasadd_readvariableop_resource:B
(conv2d_89_conv2d_readvariableop_resource:7
)conv2d_89_biasadd_readvariableop_resource:B
(conv2d_90_conv2d_readvariableop_resource:7
)conv2d_90_biasadd_readvariableop_resource:
identity¢ conv2d_87/BiasAdd/ReadVariableOp¢conv2d_87/Conv2D/ReadVariableOp¢ conv2d_88/BiasAdd/ReadVariableOp¢conv2d_88/Conv2D/ReadVariableOp¢ conv2d_89/BiasAdd/ReadVariableOp¢conv2d_89/Conv2D/ReadVariableOp¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_87/Conv2DConv2Dinputs'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
up_sampling2d_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_36/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_36/mulMulup_sampling2d_36/Const:output:0!up_sampling2d_36/Const_1:output:0*
T0*
_output_shapes
:Ò
-up_sampling2d_36/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_87/Relu:activations:0up_sampling2d_36/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
half_pixel_centers(
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
conv2d_88/Conv2DConv2D>up_sampling2d_36/resize/ResizeNearestNeighbor:resized_images:0'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides

 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88l
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88g
up_sampling2d_37/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   i
up_sampling2d_37/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_37/mulMulup_sampling2d_37/Const:output:0!up_sampling2d_37/Const_1:output:0*
T0*
_output_shapes
:Ò
-up_sampling2d_37/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_88/Relu:activations:0up_sampling2d_37/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
half_pixel_centers(
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0å
conv2d_89/Conv2DConv2D>up_sampling2d_37/resize/ResizeNearestNeighbor:resized_images:0'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides

 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppl
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppg
up_sampling2d_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   i
up_sampling2d_38/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d_38/mulMulup_sampling2d_38/Const:output:0!up_sampling2d_38/Const_1:output:0*
T0*
_output_shapes
:Ô
-up_sampling2d_38/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_89/Relu:activations:0up_sampling2d_38/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ç
conv2d_90/Conv2DConv2D>up_sampling2d_38/resize/ResizeNearestNeighbor:resized_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides

 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààt
conv2d_90/SigmoidSigmoidconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààn
IdentityIdentityconv2d_90/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÚ
NoOpNoOp!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511

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
valueB:½
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
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

%__inference_signature_wrapper_6123328
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
identity¢StatefulPartitionedCallà
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
:ÿÿÿÿÿÿÿÿÿàà*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_6122216y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
ò
ÿ
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6123732

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
 
+__inference_conv2d_87_layer_call_fn_6123647

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6122551w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
ÿ
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6123695

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
N
2__inference_up_sampling2d_37_layer_call_fn_6123700

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6122511
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
N
2__inference_max_pooling2d_37_layer_call_fn_6123603

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6122237
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6122225

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

/__inference_sequential_36_layer_call_fn_6123124

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
identity¢StatefulPartitionedCall
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_36_layer_call_and_return_conditional_losses_6122853
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
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
:ÿÿÿÿÿÿÿÿÿ88X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6123578

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

/__inference_sequential_36_layer_call_fn_6123017
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
identity¢StatefulPartitionedCall
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_36_layer_call_and_return_conditional_losses_6122953
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

_user_specified_nameimg
¿
N
2__inference_up_sampling2d_38_layer_call_fn_6123737

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6122530
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼i
¿
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123293

inputsP
6sequential_37_conv2d_84_conv2d_readvariableop_resource:E
7sequential_37_conv2d_84_biasadd_readvariableop_resource:P
6sequential_37_conv2d_85_conv2d_readvariableop_resource:E
7sequential_37_conv2d_85_biasadd_readvariableop_resource:P
6sequential_37_conv2d_86_conv2d_readvariableop_resource:E
7sequential_37_conv2d_86_biasadd_readvariableop_resource:P
6sequential_38_conv2d_87_conv2d_readvariableop_resource:E
7sequential_38_conv2d_87_biasadd_readvariableop_resource:P
6sequential_38_conv2d_88_conv2d_readvariableop_resource:E
7sequential_38_conv2d_88_biasadd_readvariableop_resource:P
6sequential_38_conv2d_89_conv2d_readvariableop_resource:E
7sequential_38_conv2d_89_biasadd_readvariableop_resource:P
6sequential_38_conv2d_90_conv2d_readvariableop_resource:E
7sequential_38_conv2d_90_biasadd_readvariableop_resource:
identity¢.sequential_37/conv2d_84/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_84/Conv2D/ReadVariableOp¢.sequential_37/conv2d_85/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_85/Conv2D/ReadVariableOp¢.sequential_37/conv2d_86/BiasAdd/ReadVariableOp¢-sequential_37/conv2d_86/Conv2D/ReadVariableOp¢.sequential_38/conv2d_87/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_87/Conv2D/ReadVariableOp¢.sequential_38/conv2d_88/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_88/Conv2D/ReadVariableOp¢.sequential_38/conv2d_89/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_89/Conv2D/ReadVariableOp¢.sequential_38/conv2d_90/BiasAdd/ReadVariableOp¢-sequential_38/conv2d_90/Conv2D/ReadVariableOp¬
-sequential_37/conv2d_84/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_84_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ë
sequential_37/conv2d_84/Conv2DConv2Dinputs5sequential_37/conv2d_84/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¢
.sequential_37/conv2d_84/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_84_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
sequential_37/conv2d_84/BiasAddBiasAdd'sequential_37/conv2d_84/Conv2D:output:06sequential_37/conv2d_84/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
sequential_37/conv2d_84/SigmoidSigmoid(sequential_37/conv2d_84/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààÃ
&sequential_37/max_pooling2d_36/MaxPoolMaxPool#sequential_37/conv2d_84/Sigmoid:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
ksize
*
paddingVALID*
strides
¬
-sequential_37/conv2d_85/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_85_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_37/conv2d_85/Conv2DConv2D/sequential_37/max_pooling2d_36/MaxPool:output:05sequential_37/conv2d_85/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¢
.sequential_37/conv2d_85/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_85_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_37/conv2d_85/BiasAddBiasAdd'sequential_37/conv2d_85/Conv2D:output:06sequential_37/conv2d_85/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
sequential_37/conv2d_85/ReluRelu(sequential_37/conv2d_85/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppÊ
&sequential_37/max_pooling2d_37/MaxPoolMaxPool*sequential_37/conv2d_85/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
ksize
*
paddingVALID*
strides
¬
-sequential_37/conv2d_86/Conv2D/ReadVariableOpReadVariableOp6sequential_37_conv2d_86_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_37/conv2d_86/Conv2DConv2D/sequential_37/max_pooling2d_37/MaxPool:output:05sequential_37/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¢
.sequential_37/conv2d_86/BiasAdd/ReadVariableOpReadVariableOp7sequential_37_conv2d_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_37/conv2d_86/BiasAddBiasAdd'sequential_37/conv2d_86/Conv2D:output:06sequential_37/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
sequential_37/conv2d_86/ReluRelu(sequential_37/conv2d_86/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88Ê
&sequential_37/max_pooling2d_38/MaxPoolMaxPool*sequential_37/conv2d_86/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
¬
-sequential_38/conv2d_87/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_87_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ò
sequential_38/conv2d_87/Conv2DConv2D/sequential_37/max_pooling2d_38/MaxPool:output:05sequential_38/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¢
.sequential_38/conv2d_87/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_87/BiasAddBiasAdd'sequential_38/conv2d_87/Conv2D:output:06sequential_38/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_38/conv2d_87/ReluRelu(sequential_38/conv2d_87/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$sequential_38/up_sampling2d_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"      w
&sequential_38/up_sampling2d_36/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_36/mulMul-sequential_38/up_sampling2d_36/Const:output:0/sequential_38/up_sampling2d_36/Const_1:output:0*
T0*
_output_shapes
:ü
;sequential_38/up_sampling2d_36/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_87/Relu:activations:0&sequential_38/up_sampling2d_36/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
half_pixel_centers(¬
-sequential_38/conv2d_88/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_88_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_88/Conv2DConv2DLsequential_38/up_sampling2d_36/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*
paddingSAME*
strides
¢
.sequential_38/conv2d_88/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_88_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_88/BiasAddBiasAdd'sequential_38/conv2d_88/Conv2D:output:06sequential_38/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
sequential_38/conv2d_88/ReluRelu(sequential_38/conv2d_88/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88u
$sequential_38/up_sampling2d_37/ConstConst*
_output_shapes
:*
dtype0*
valueB"8   8   w
&sequential_38/up_sampling2d_37/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_37/mulMul-sequential_38/up_sampling2d_37/Const:output:0/sequential_38/up_sampling2d_37/Const_1:output:0*
T0*
_output_shapes
:ü
;sequential_38/up_sampling2d_37/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_88/Relu:activations:0&sequential_38/up_sampling2d_37/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
half_pixel_centers(¬
-sequential_38/conv2d_89/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_89_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_89/Conv2DConv2DLsequential_38/up_sampling2d_37/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
paddingSAME*
strides
¢
.sequential_38/conv2d_89/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_38/conv2d_89/BiasAddBiasAdd'sequential_38/conv2d_89/Conv2D:output:06sequential_38/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
sequential_38/conv2d_89/ReluRelu(sequential_38/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppu
$sequential_38/up_sampling2d_38/ConstConst*
_output_shapes
:*
dtype0*
valueB"p   p   w
&sequential_38/up_sampling2d_38/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ®
"sequential_38/up_sampling2d_38/mulMul-sequential_38/up_sampling2d_38/Const:output:0/sequential_38/up_sampling2d_38/Const_1:output:0*
T0*
_output_shapes
:þ
;sequential_38/up_sampling2d_38/resize/ResizeNearestNeighborResizeNearestNeighbor*sequential_38/conv2d_89/Relu:activations:0&sequential_38/up_sampling2d_38/mul:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(¬
-sequential_38/conv2d_90/Conv2D/ReadVariableOpReadVariableOp6sequential_38_conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
sequential_38/conv2d_90/Conv2DConv2DLsequential_38/up_sampling2d_38/resize/ResizeNearestNeighbor:resized_images:05sequential_38/conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
paddingSAME*
strides
¢
.sequential_38/conv2d_90/BiasAdd/ReadVariableOpReadVariableOp7sequential_38_conv2d_90_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
sequential_38/conv2d_90/BiasAddBiasAdd'sequential_38/conv2d_90/Conv2D:output:06sequential_38/conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
sequential_38/conv2d_90/SigmoidSigmoid(sequential_38/conv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà|
IdentityIdentity#sequential_38/conv2d_90/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààí
NoOpNoOp/^sequential_37/conv2d_84/BiasAdd/ReadVariableOp.^sequential_37/conv2d_84/Conv2D/ReadVariableOp/^sequential_37/conv2d_85/BiasAdd/ReadVariableOp.^sequential_37/conv2d_85/Conv2D/ReadVariableOp/^sequential_37/conv2d_86/BiasAdd/ReadVariableOp.^sequential_37/conv2d_86/Conv2D/ReadVariableOp/^sequential_38/conv2d_87/BiasAdd/ReadVariableOp.^sequential_38/conv2d_87/Conv2D/ReadVariableOp/^sequential_38/conv2d_88/BiasAdd/ReadVariableOp.^sequential_38/conv2d_88/Conv2D/ReadVariableOp/^sequential_38/conv2d_89/BiasAdd/ReadVariableOp.^sequential_38/conv2d_89/Conv2D/ReadVariableOp/^sequential_38/conv2d_90/BiasAdd/ReadVariableOp.^sequential_38/conv2d_90/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿàà: : : : : : : : : : : : : : 2`
.sequential_37/conv2d_84/BiasAdd/ReadVariableOp.sequential_37/conv2d_84/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_84/Conv2D/ReadVariableOp-sequential_37/conv2d_84/Conv2D/ReadVariableOp2`
.sequential_37/conv2d_85/BiasAdd/ReadVariableOp.sequential_37/conv2d_85/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_85/Conv2D/ReadVariableOp-sequential_37/conv2d_85/Conv2D/ReadVariableOp2`
.sequential_37/conv2d_86/BiasAdd/ReadVariableOp.sequential_37/conv2d_86/BiasAdd/ReadVariableOp2^
-sequential_37/conv2d_86/Conv2D/ReadVariableOp-sequential_37/conv2d_86/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_87/BiasAdd/ReadVariableOp.sequential_38/conv2d_87/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_87/Conv2D/ReadVariableOp-sequential_38/conv2d_87/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_88/BiasAdd/ReadVariableOp.sequential_38/conv2d_88/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_88/Conv2D/ReadVariableOp-sequential_38/conv2d_88/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_89/BiasAdd/ReadVariableOp.sequential_38/conv2d_89/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_89/Conv2D/ReadVariableOp-sequential_38/conv2d_89/Conv2D/ReadVariableOp2`
.sequential_38/conv2d_90/BiasAdd/ReadVariableOp.sequential_38/conv2d_90/BiasAdd/ReadVariableOp2^
-sequential_38/conv2d_90/Conv2D/ReadVariableOp-sequential_38/conv2d_90/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6122288

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp*
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
:ÿÿÿÿÿÿÿÿÿppX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿppw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿpp: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿpp
 
_user_specified_nameinputs
ñ
 
+__inference_conv2d_86_layer_call_fn_6123617

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6122306w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ88
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6122249

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
=
img6
serving_default_img:0ÿÿÿÿÿÿÿÿÿààK
sequential_38:
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿààtensorflow/serving/predict:î
¾
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
Æ
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
í
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
ë
&iter

'beta_1

(beta_2
	)decay
*learning_rate+mã,mä-må.mæ/mç0mè1mé2mê3më4mì5mí6mî7mï8mð+vñ,vò-vó.vô/võ0vö1v÷2vø3vù4vú5vû6vü7vý8vþ"
	optimizer
 "
trackable_list_wrapper

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

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
Ê
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
2
/__inference_sequential_36_layer_call_fn_6122884
/__inference_sequential_36_layer_call_fn_6123124
/__inference_sequential_36_layer_call_fn_6123157
/__inference_sequential_36_layer_call_fn_6123017À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123225
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123293
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123051
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123085À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÉBÆ
"__inference__wrapped_model_6122216img"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
>serving_default"
signature_map
»

+kernel
,bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
»

-kernel
.bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
»

/kernel
0bias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
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
­
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
2
/__inference_sequential_37_layer_call_fn_6122329
/__inference_sequential_37_layer_call_fn_6123345
/__inference_sequential_37_layer_call_fn_6123362
/__inference_sequential_37_layer_call_fn_6122432À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123390
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123418
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122454
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122476À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
»

1kernel
2bias
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
»

3kernel
4bias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

5kernel
6bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

7kernel
8bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_sequential_38_layer_call_fn_6122631
/__inference_sequential_38_layer_call_fn_6123439
/__inference_sequential_38_layer_call_fn_6123460
/__inference_sequential_38_layer_call_fn_6122761À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123504
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123548
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122788
J__inference_sequential_38_layer_call_and_return_conditional_losses_6122815À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv2d_84/kernel
:2conv2d_84/bias
*:(2conv2d_85/kernel
:2conv2d_85/bias
*:(2conv2d_86/kernel
:2conv2d_86/bias
*:(2conv2d_87/kernel
:2conv2d_87/bias
*:(2conv2d_88/kernel
:2conv2d_88/bias
*:(2conv2d_89/kernel
:2conv2d_89/bias
*:(2conv2d_90/kernel
:2conv2d_90/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÈBÅ
%__inference_signature_wrapper_6123328img"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_84_layer_call_fn_6123557¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6123568¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_36_layer_call_fn_6123573¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6123578¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_85_layer_call_fn_6123587¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6123598¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_37_layer_call_fn_6123603¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6123608¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_86_layer_call_fn_6123617¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6123628¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_max_pooling2d_38_layer_call_fn_6123633¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6123638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_87_layer_call_fn_6123647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6123658¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_up_sampling2d_36_layer_call_fn_6123663¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6123675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_88_layer_call_fn_6123684¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_88_layer_call_and_return_conditional_losses_6123695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_up_sampling2d_37_layer_call_fn_6123700¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6123712¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_89_layer_call_fn_6123721¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_89_layer_call_and_return_conditional_losses_6123732¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
2__inference_up_sampling2d_38_layer_call_fn_6123737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6123749¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_conv2d_90_layer_call_fn_6123758¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_90_layer_call_and_return_conditional_losses_6123769¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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

Útotal

Ûcount
Ü	variables
Ý	keras_api"
_tf_keras_metric
c

Þtotal

ßcount
à
_fn_kwargs
á	variables
â	keras_api"
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
Ú0
Û1"
trackable_list_wrapper
.
Ü	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Þ0
ß1"
trackable_list_wrapper
.
á	variables"
_generic_user_object
/:-2Adam/conv2d_84/kernel/m
!:2Adam/conv2d_84/bias/m
/:-2Adam/conv2d_85/kernel/m
!:2Adam/conv2d_85/bias/m
/:-2Adam/conv2d_86/kernel/m
!:2Adam/conv2d_86/bias/m
/:-2Adam/conv2d_87/kernel/m
!:2Adam/conv2d_87/bias/m
/:-2Adam/conv2d_88/kernel/m
!:2Adam/conv2d_88/bias/m
/:-2Adam/conv2d_89/kernel/m
!:2Adam/conv2d_89/bias/m
/:-2Adam/conv2d_90/kernel/m
!:2Adam/conv2d_90/bias/m
/:-2Adam/conv2d_84/kernel/v
!:2Adam/conv2d_84/bias/v
/:-2Adam/conv2d_85/kernel/v
!:2Adam/conv2d_85/bias/v
/:-2Adam/conv2d_86/kernel/v
!:2Adam/conv2d_86/bias/v
/:-2Adam/conv2d_87/kernel/v
!:2Adam/conv2d_87/bias/v
/:-2Adam/conv2d_88/kernel/v
!:2Adam/conv2d_88/bias/v
/:-2Adam/conv2d_89/kernel/v
!:2Adam/conv2d_89/bias/v
/:-2Adam/conv2d_90/kernel/v
!:2Adam/conv2d_90/bias/v¸
"__inference__wrapped_model_6122216+,-./0123456786¢3
,¢)
'$
imgÿÿÿÿÿÿÿÿÿàà
ª "GªD
B
sequential_381.
sequential_38ÿÿÿÿÿÿÿÿÿààº
F__inference_conv2d_84_layer_call_and_return_conditional_losses_6123568p+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 
+__inference_conv2d_84_layer_call_fn_6123557c+,9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª ""ÿÿÿÿÿÿÿÿÿàà¶
F__inference_conv2d_85_layer_call_and_return_conditional_losses_6123598l-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿpp
 
+__inference_conv2d_85_layer_call_fn_6123587_-.7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿpp
ª " ÿÿÿÿÿÿÿÿÿpp¶
F__inference_conv2d_86_layer_call_and_return_conditional_losses_6123628l/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ88
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ88
 
+__inference_conv2d_86_layer_call_fn_6123617_/07¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ88
ª " ÿÿÿÿÿÿÿÿÿ88¶
F__inference_conv2d_87_layer_call_and_return_conditional_losses_6123658l127¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_87_layer_call_fn_6123647_127¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÛ
F__inference_conv2d_88_layer_call_and_return_conditional_losses_612369534I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
+__inference_conv2d_88_layer_call_fn_612368434I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
F__inference_conv2d_89_layer_call_and_return_conditional_losses_612373256I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
+__inference_conv2d_89_layer_call_fn_612372156I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÛ
F__inference_conv2d_90_layer_call_and_return_conditional_losses_612376978I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
+__inference_conv2d_90_layer_call_fn_612375878I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_36_layer_call_and_return_conditional_losses_6123578R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_36_layer_call_fn_6123573R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_37_layer_call_and_return_conditional_losses_6123608R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_37_layer_call_fn_6123603R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_38_layer_call_and_return_conditional_losses_6123638R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_38_layer_call_fn_6123633R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿà
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123051+,-./012345678>¢;
4¢1
'$
imgÿÿÿÿÿÿÿÿÿàà
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 à
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123085+,-./012345678>¢;
4¢1
'$
imgÿÿÿÿÿÿÿÿÿàà
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ó
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123225+,-./012345678A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 Ó
J__inference_sequential_36_layer_call_and_return_conditional_losses_6123293+,-./012345678A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 ¸
/__inference_sequential_36_layer_call_fn_6122884+,-./012345678>¢;
4¢1
'$
imgÿÿÿÿÿÿÿÿÿàà
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
/__inference_sequential_36_layer_call_fn_6123017+,-./012345678>¢;
4¢1
'$
imgÿÿÿÿÿÿÿÿÿàà
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
/__inference_sequential_36_layer_call_fn_6123124+,-./012345678A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
/__inference_sequential_36_layer_call_fn_6123157+,-./012345678A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122454+,-./0J¢G
@¢=
30
conv2d_84_inputÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ò
J__inference_sequential_37_layer_call_and_return_conditional_losses_6122476+,-./0J¢G
@¢=
30
conv2d_84_inputÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123390z+,-./0A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 È
J__inference_sequential_37_layer_call_and_return_conditional_losses_6123418z+,-./0A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ©
/__inference_sequential_37_layer_call_fn_6122329v+,-./0J¢G
@¢=
30
conv2d_84_inputÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ©
/__inference_sequential_37_layer_call_fn_6122432v+,-./0J¢G
@¢=
30
conv2d_84_inputÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_37_layer_call_fn_6123345m+,-./0A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 

 
ª " ÿÿÿÿÿÿÿÿÿ 
/__inference_sequential_37_layer_call_fn_6123362m+,-./0A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿàà
p

 
ª " ÿÿÿÿÿÿÿÿÿä
J__inference_sequential_38_layer_call_and_return_conditional_losses_612278812345678H¢E
>¢;
1.
conv2d_87_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ä
J__inference_sequential_38_layer_call_and_return_conditional_losses_612281512345678H¢E
>¢;
1.
conv2d_87_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123504|12345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 Ê
J__inference_sequential_38_layer_call_and_return_conditional_losses_6123548|12345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 ¼
/__inference_sequential_38_layer_call_fn_612263112345678H¢E
>¢;
1.
conv2d_87_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
/__inference_sequential_38_layer_call_fn_612276112345678H¢E
>¢;
1.
conv2d_87_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
/__inference_sequential_38_layer_call_fn_612343912345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
/__inference_sequential_38_layer_call_fn_612346012345678?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
%__inference_signature_wrapper_6123328+,-./012345678=¢:
¢ 
3ª0
.
img'$
imgÿÿÿÿÿÿÿÿÿàà"GªD
B
sequential_381.
sequential_38ÿÿÿÿÿÿÿÿÿààð
M__inference_up_sampling2d_36_layer_call_and_return_conditional_losses_6123675R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_up_sampling2d_36_layer_call_fn_6123663R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_up_sampling2d_37_layer_call_and_return_conditional_losses_6123712R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_up_sampling2d_37_layer_call_fn_6123700R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_up_sampling2d_38_layer_call_and_return_conditional_losses_6123749R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_up_sampling2d_38_layer_call_fn_6123737R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ