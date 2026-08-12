within {PKG}.Layers;
function relu "Elementwise rectified linear unit"
  input Real x[:];
  output Real y[size(x, 1)];
algorithm
  for i in 1:size(x, 1) loop
    y[i] := noEvent(if x[i] > 0 then x[i] else 0);
  end for;
end relu;
