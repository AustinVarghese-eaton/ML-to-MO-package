within {PKG}.Layers;
function identity "Identity passthrough"
  input Real x[:];
  output Real y[size(x, 1)];
algorithm
  y := x;
end identity;
