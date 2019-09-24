using Random
using LinearAlgebra
using Plots
"""
    problem_data()

Returns the Q, q, a, and b matrix/vectors that defines the problem in Hand-In 1.

"""
k=30000
function problem_data()
	mt = MersenneTwister(123)

	n = 20

	Qv = randn(mt,n,n)
	Q = Qv'*Qv
	q = randn(mt,n)

	a = -rand(mt,n)
	b = rand(mt,n)

	return Q,q,a,b
end

#Task 6.
function primal_prob(Q,q,a,b, gamma)
	n = 20
	x = zeros(Float64,n,k+1)
	x= randn(n,k+1)
	for i=2:k+1
		x[:,i] = prox_box(x[:,i-1] - gamma*grad_quad(x[:,i-1], Q, q),a,b,gamma)
	end
	return x
end

#Task 7.
function dual_prob(Q,q,a,b,gamma)
	n = 20
	y = zeros(Float64,n,k+1)
	for i=2:k+1
		y[:,i] = -prox_boxconj(-y[:,i-1] + gamma*grad_quadconj(y[:,i-1], Q, q),a,b,gamma)
	end
	return dual2primal(y,Q,q,a,b), y
end

#Task 6 helper. (Will save the plots in current directory.)
function run_1(Q,q,a,b,g)
	gamma = g/opnorm(Q)
	res_x = primal_prob(Q,q,a,b, gamma)
	_, length = size(res_x)
	diff = res_x[:,2:length] .- res_x[:,1:length-1]
	norm_diff = sqrt.(sum(abs2, diff, dims=1))
	fig = plot(norm_diff', legend=false,
		xlabel = "iteration nbr",
		title = "gamma = $g/L",
		yscale=:log10)
	savefig("primal_gamma$g.png")
	return res_x
end

#Task 7 helper.
function run_2(Q,q,a,b,g)
	gamma = g/opnorm(inv(Q))
	res_dual, y = dual_prob(Q,q,a,b, gamma)
	_, length = size(y)
	diff = y[:,2:length] .- y[:,1:length-1]
	norm_diff = sqrt.(sum(abs2, diff, dims=1))
	fig = plot(norm_diff', legend=false,
		xlabel = "iteration nbr",
		title = "gamma = $g/L*",
		yscale=:log10)
	savefig("dual_gamma$g.png")
	return res_dual,y
end

#Task 7, will create a plot with functionvalue.
function dual_quads(xd,Q,q)
	_, length = size(xd)
	a = zeros(Float64, length)
	for i=1: length
		a[i] = quad(xd[:,i], Q, q)
	end
	fig = plot(a, legend=false, xlabel = "iteration nbr", ylabel="function value")
	savefig("dual_function_values.png")
	return a
end
