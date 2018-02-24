



disp(""); 
disp("======== initial network =========");

disp("");
disp("biases of first layer");
B1=[-0.553; -0.504; 0.813; -0.386]

disp("");
disp("weights of first layer");
W1=[-0.881,  0.930, -0.382, -0.652;
    -0.842,  0.579, -0.610, -0.311;
    -0.422, -0.711, -0.348,  0.150;
    -0.069,  0.482,  0.275, -0.242]
    
disp("");
disp("biases of second layer");
B2=[-0.883; 0.103]

disp("");
disp("weights of second layer");
W2 = [-0.961,  0.450, -0.442, -0.499;
      -0.284,  0.899,  0.679, -0.614]

          
      
     
disp("");
disp(""); 
disp("======== evaluationg =========");

disp("");
disp("input is:");
%X=[1; 1; 0; 0]
X=[0; 1; 0; 1]

disp("");
disp("z1 is:");
Z1=(B1 + W1*X)

disp("");
disp("a1 is:");
A1=arrayfun("sigmoid", Z1)

disp("");
disp("z2 is:");
Z2=(B2 + W2*A1)

disp("");
disp("a2 is:");
A2=arrayfun("sigmoid", Z2)




disp("");
disp("");
disp("======== gradient =========");

disp("");
disp("expected output: ");
%Y=[1; 0]
Y=[0; 1]

disp("");
disp("for each neuron of SECOND layer, ");
disp("derivative of error function with respect to...");

disp("");
disp(" ... its weighted sum: ");
G2=(A2-Y).*arrayfun("sigmoid_prime", Z2)

disp("");
disp(" ... its bias: ")
GB2 = G2

disp("");
disp(" ... its weights: ");
GW2 = G2*transpose(A1)




disp("");
disp("for each neuron of FIRST layer, ");
disp("derivative of error function with respect to...")

disp("");
disp(" ... its weighted sum: ");
G1 = ( transpose(W2)*G2 ) .* arrayfun("sigmoid_prime", Z1)

disp("");
disp(" ... its bias: ")
GB1 = G1

disp("");
disp(" ... its weights: ");
GW1 = G1*transpose(X)


disp("");
disp("======== ad hoc =========");

GB1
GW1
GB2
GW2