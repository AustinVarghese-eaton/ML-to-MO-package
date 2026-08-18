within {PKG}.Layers;
function exp_mask "Invert log1p on masked entries: mask>0.5 ? exp(x)-1 : x"
  input Real x[:];
  input Real mask[size(x, 1)];
  output Real y[size(x, 1)];
algorithm
  for i in 1:size(x, 1) loop
    y[i] := if mask[i] > 0.5 then Modelica.Math.exp(x[i]) - 1.0 else x[i];
  end for;
end exp_mask;
