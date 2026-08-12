within {PKG}.Layers;
function affine_scale "Standardize inputs: (x - mean)/scale"
  input Real x[:];
  input Real mean[size(x, 1)];
  input Real scale[size(x, 1)];
  output Real y[size(x, 1)];
protected
  constant Real eps = 1e-12;
  Real denom;
algorithm
  for i in 1:size(x, 1) loop
    denom := noEvent(if abs(scale[i]) > eps then scale[i] else 1.0);
    y[i] := (x[i] - mean[i])/denom;
  end for;
end affine_scale;
