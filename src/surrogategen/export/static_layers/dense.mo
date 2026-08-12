within {PKG}.Layers;
function dense "Affine layer: y[i] = b[i] + sum_j W[i,j]*x[j]"
  input Real x[:];
  input Real W[:, :];
  input Real b[:];
  output Real y[size(W, 1)];
algorithm
  for i in 1:size(W, 1) loop
    y[i] := b[i] + sum(W[i, j]*x[j] for j in 1:size(W, 2));
  end for;
end dense;
