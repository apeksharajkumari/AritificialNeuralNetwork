input=6;
hidden=4; 
output=6; 
c=1; //sigmoid constant

//defining the sigmoid function

function [sg]=sigmoid(c, x)
xlen=length(x);
unitmatrix=ones(1,xlen);
sg=(unitmatrix'+exp(-c*x));
sg=unitmatrix'./sg;
endfunction

//function for changing w2 vector

function [dw]=dw2(err,w2)
error=[];
sz=size(w2);
hidden=sz(1,2) ;
for i=1:hidden
error=[error,err] ; //to match dimension with w2
end
dw=w2.*error ;
endfunction

//assign random weights at first

w1=rand(hidden, input)
w2=rand(output, hidden) 

inputvector=[3 4 5 1 2 5]' //input vector
desiredout=[1 1 1 1 1 1]' //desired out vector

neth=w1*inputvector //net at hidden nodes
hout=sigmoid(1,neth)  //net output from hidden nodes
neto=w2*hout  //net at output nodes
actualout=sigmoid(1,neto)  //actual out vector
err=desiredout-actualout //error vector

for i=1:40 //40 itertations for reducing error
cdw2=dw2(err,w2); //change in w2 vector
rscdw2=sum(cdw2,1); //calculate rowsum 
trs=[];

for i=1:input
trs=[trs,rscdw2']; //to match dimension
end

tiv=[]; //total input vector
for i=1:hidden
tiv=[tiv;inputvector']; //to match dimension
end

dw1=tiv.*trs;
w1=w1+dw1; //new w1 matrix
w2=w2+cdw2; //new w2 matrix

neth=w1*inputvector; 
hout =sigmoid(1,neth) ; 
neto =w2*hout ; 
actualout=sigmoid(1,neto); 
err=desiredout-actualout; //error vector

end;

// FINAL VALUES

neth=w1*inputvector; 
hout=sigmoid(1,neth); 
neto=w2*hout; 
actualout=sigmoid(1,neto); //output vector
inputvector
desiredout
actualout 
