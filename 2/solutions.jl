using LinearAlgebra
using ProximalOperators
using Plots

function reg_least_squares(p , lambda, lasso=false)
    x, y = leastsquares_data()
    x_scaled = scaling(x)
	#y_scaled = scaling(y)
	X = gen_features(x_scaled, p)
	step_size = 1 / opnorm(X*X') # or max eigenvalue
	#Iterate until no longer changing.
	k = 1000
	w = zeros(p+1, k+1)

	if lasso
		g = NormL1(lambda)
	else
		g = SqrNormL2(2*lambda)
	end
	f = SqrNormL2(1)

	for i=2:k+1
		grad, _ = gradient(f, (X'*w[:,i-1] .- y))
		w[:, i], _ = prox(g, (w[:,i-1] - step_size*X*grad),step_size)
	end
	return w
end

function model_output(x, w)
	n = length(w)
	y = similar(x)
	#Maybe vectorize this in a nice way.
	for k=1:length(x)
		m = 0
		for i=1:n
			m += x[k]^(i-1)*w[i]
		end
		y[k] = m
	end
	return y
end

function plots(w)
	r = range(-1,stop=1, length=100)
	out = model_output(r,w)
	println(out)
	x, y = leastsquares_data()
	x_scaled = scaling(x)
	fig = plot(r,out)
end

function gen_features(x_scaled, p)
	features = zeros(Float64, p+1, length(x_scaled))
	features[1,:] .= 1
	for i=2: p+1
		features[i,:] = x_scaled' .^ (i-1)
	end
	return features
end

#min max scaler.
function scaling(data)
    #r(X) = (x-B)*sigma.
    data_range = maximum(data) - minimum(data)
    m01 = (data .- minimum(data)) ./ data_range
    m = 2 .* m01 .- 1
end
