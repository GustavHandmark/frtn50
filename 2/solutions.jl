using LinearAlgebra
using ProximalOperators
using Plots

function reg_least_squares(p , lambda, lasso=false, scale=true)
    x, y = leastsquares_data()
	if scale
		x_scaled = scaling(x)
	else
		x_scaled = x
	end
	X = gen_features(x_scaled, p)  qwertyuiopå

	step_size = 1 / opnorm(X*X') # or max eigenvalue
	#Iterate until no longer changing.
	k = 1000000
	w = zeros(p+1, k+1)

	if lasso
		g = NormL1(lambda)
	else
		g = SqrNormL2(2*lambda)
	end
	f = SqrNormL2(1)
	for i=2:k+1
		grad, _ = gradient(f, (X'*w[:,i-1] .- y))
		w[:, i], _ = prox(g, (w[:,i-1] - step_size*X*grad), step_size)
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

function plots(w, p, title, scale = true)
	x, y = leastsquares_data()
	r = range(-1,stop=3, length=100)
	out = model_output(r,w)
	if scale
		x_scaled=scaling(x)
	else
		x_scaled = x
	end
	fig = plot(r,out, legend = false,
		xlabel="x",
		ylabel="y",
		title = title)
	plot!(x_scaled, y, seriestype=:scatter)
	file_name = "Least_squares_$title"
	file_name = replace(file_name,"." => "_")
	savefig(file_name * ".png")
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

function task_2()
	lambda = 0
	for p=1:10
		w = reg_least_squares(p, lambda)
		plots(w[:,end], p, "p=$p")
		println(w[:,end])
		println("Norm final w.")
		println(sqrt.(sum(abs2, w[:,end], dims=1)))
	end
end

function task_3()
	p = 10
	lasso = false #q =2
	lambda = exp10.(range(-3, stop=1, length=10))
	for l in lambda
		l = round(l, digits=4)
		w = reg_least_squares(p, l, lasso)
		plots(w[:,end], p, "lambda=$l")
		println(w[:,end])
		println("Norm final w.")
		println(sqrt.(sum(abs2, w[:,end], dims=1)))
	end
end
function task_3b()
	p = 10
	lasso = true #q =1
	lambda = exp10.(range(-3, stop=1, length=10))
	for l in lambda
		l = round(l, digits=4)
		w = reg_least_squares(p, l, lasso)
		plots(w[:,end], p, "Lasso. lambda=$l")
		println(w[:,end])
		println("Norm final w.")
		println(sqrt.(sum(abs2, w[:,end], dims=1)))
	end
end

function task_4()
	p = 10
	lasso = false
	lambda = 0.06
	w = reg_least_squares(p, lambda, lasso, false) # no scaling
	plots(w[:,end], p, "q=2, p=10, lambda=$lambda", false)
	println(w[:,end])
	println("Norm final w.")
	println(sqrt.(sum(abs2, w[:,end], dims=1)))
end
