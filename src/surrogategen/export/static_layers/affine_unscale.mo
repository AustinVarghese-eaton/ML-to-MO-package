within {PKG}.Layers;
function affine_unscale "Un-standardize outputs: x*scale + mean"
  input Real x[:];
  input Real mean[size(x, 1)];
  input Real scale[size(x, 1)];
  output Real y[size(x, 1)];
algorithm
  for i in 1:size(x, 1) loop
    y[i] := x[i]*scale[i] + mean[i];
  end for;
end affine_unscale;
