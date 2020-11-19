#include "mpc.h"


Mpcc::Mpcc() {};

Mpcc::Mpcc(std::vector<std::vector<double>> AMatrix, 
    std::vector<std::vector<double>> BMatrix,
    std::vector<std::vector<double>> QMatrix,
    std::vector<std::vector<double>> RMatrix,
    std::vector<double> lowerbound_x,
    std::vector<double> upperbound_x,
    std::vector<double> lowerbound_u,
    std::vector<double> upperbound_u,
    std::vector<double> target,
    size_t horizon)
    :n(AMatrix.size()),
    p(BMatrix[0].size()),
    lb_x(lowerbound_x),
    ub_x(upperbound_x),
    lb_u(lowerbound_u),
    ub_u(upperbound_u),
    target(target),
    N(horizon),
    x(SX::sym("x", n, 1)),
    ref(SX::sym("ref", n, 1)),
    u(SX::sym("u", p, 1)),
    A(SX::sym("A", n, n)),
    B(SX::sym("B", n, p)),
    Q(SX::sym("Q", n, n)),
    R(SX::sym("R", p, p)),
    X(SX::sym("X", (N+1)*n, 1)),
    REF(SX::sym("REF", (N+1)*n, 1)),
    U(SX::sym("U", N*p, 1)),
    firstU(X.size1()),
    state(n),
    action(p),
    actions(p, std::vector<double>(horizon)){
    // Convert vector of vectors to Casadi object
    for (int i=0;i<n;++i){ // row
        for (int j=0;j<n;++j){ // col
            A(i,j) = AMatrix[i][j];
        }
    }
    // Convert vector of vectors to Casadi object
    for (int i=0;i<n;++i){ // row
        for(int j=0;j<p;++j){ // col
            B(i,j) = BMatrix[i][j];
        }
    }
    // Create system equation
    xDot = mtimes(A, x) + mtimes(B, u);

    // create function to calculate xDot
    system = Function("sys", {x, u}, {xDot});

    for (int i=0;i<n;++i){ // row
        for (int j=0;j<n;++j){ // col
            Q(i,j) = QMatrix[i][j];
        }
    }
    for (int i=0;i<p;++i){ // row
        for (int j=0;j<p;++j){ // col
            R(i,j) = RMatrix[i][j];
        }
    }
    // (x)^T * Q * (x)      +       u^T * R * u
    stage_cost = mtimes(mtimes(transpose(x-ref),Q),(x-ref)) + 
                    mtimes(mtimes(transpose(u), R), u);
    stage_cost_fcn = Function("stage_cost", {x, u, ref}, {stage_cost});
    terminal_cost = mtimes(mtimes(transpose(x-ref),Q),(x-ref));
    terminal_cost_fcn = Function("terminal_cost", {x, ref}, {terminal_cost});

    x_k = SX::zeros(n);
    x_k_next = SX::zeros(n);
    x_k_next_calc = SX::zeros(n);
    u_k = SX::zeros(p);
    ref_k = SX::zeros(n);
    J = SX(0);
    zeroVec = std::vector<double>(n, 0);
    for (size_t k=0; k < N; ++k){
        for (size_t j= 0; j < n; ++j){
            // read current vector in x_k
            x_k(j) = X(k*n + j);
            // shift index by n -> read next vector 
            x_k_next(j) = X((k+1)*n + j);
            // connect ref
            ref_k(j) = REF(k*n + j);
        }
        for (size_t m = 0; m < p; ++m){
            // read current u vector in u
            u_k(m) = U(k*p + m);
        }
        // Add cost of current stage to stage cost
        J += stage_cost_fcn(std::vector<SX>{x_k, u_k, ref_k}).at(0);
        // Evaluate system symbolically (equality constraint)
        // Important: the x of the system is the difference betwee
        // the actual x and und desired ref
        x_k_next_calc  = system(std::vector<SX>{x_k - ref_k, u_k}).at(0);
        // Add equality constraint to g ...
        // The optimizer will try to make this equal to zero
        g.push_back(x_k_next - x_k_next_calc -ref_k);
        // ... and update lb_g ub_g by added a zero for each element
        for (size_t i=0; i < n; ++i){
            lb_g.push_back(0);
            ub_g.push_back(0);
        }
        // same for bounds of x and u. Append to large vector
        for(size_t i=0; i < n; ++i){
            lb_X.push_back(lb_x[i]);
            ub_X.push_back(ub_x[i]);
        }
        for(size_t i=0; i < p; ++i){
            lb_U.push_back(lb_u[i]);
            ub_U.push_back(ub_u[i]);
        }

    }
    // Finally, the terminal cost and constraint
    x_terminal = SX::zeros(n);
    for (size_t i = 0; i < n; ++i){
        x_terminal(i) = X(N*n + i);
        ref_k(i) = REF(N*n + i);
    }
    J += terminal_cost_fcn(std::vector<SX>{x_terminal, ref_k}).at(0);
    // add constraints to vector
    for(size_t i=0; i < n; ++i){
        lb_X.push_back(lb_x[i]);
        ub_X.push_back(ub_x[i]);
    }
    // copy from target vector 
    for (size_t i=0; i < N + 1; ++i){
        for (size_t j=0; j < n; ++j){
            //targetTrajectory[i*n + j] = target[j];
            targetTrajectory.push_back(target[j]);
        }
    }

    //// Create NLP solver
    lbx = SX::vertcat({lb_X, lb_U});
    ubx = SX::vertcat({ub_X, ub_U});
    xx = SX::vertcat({X,U});
    lbg = SX::vertcat({lb_g});
    ubg = SX::vertcat({ub_g});
    gg = SX::vertcat({g});
    opts_dict=Dict();
    opts_dict["ipopt.print_level"] = 0;
    opts_dict["ipopt.sb"] = "yes";
    opts_dict["print_time"] = 0;
    prob = {{"f", J}, {"x", xx}, {"g", gg}, {"p", REF}};
    mpc_solver = nlpsol("solver", "ipopt", prob, opts_dict);

    // Set up boundaries for mpc loop
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
}

std::vector<double> Mpcc::get_all_controls(const std::vector<double>& currentState){
        // Update start constraints
        for (size_t i=0; i < n; ++i){
            lbx(i) = SX(currentState[i]);
            ubx(i) = SX(currentState[i]);
        }
        arg["lbx"] = lbx;
        arg["ubx"] = ubx;
        arg["p"] = targetTrajectory;
        // Solve optimization problem
        mpc_res = mpc_solver(arg);
        // tmp will contain computed x AND u
        tmp = std::vector<double>(mpc_res["x"]);
        // Return computed u from the result. This will be of size N
        return std::vector<double>(tmp.end() - N*p, tmp.end());
    }

void Mpcc::updateTarget(const std::vector<double>& newTarget){
        targetTrajectory.clear();
        // copy from target vector 
        for (size_t i=0; i < N + 1; ++i){
            for (size_t j=0; j < n; ++j){
                targetTrajectory.push_back(newTarget[j]);
            }
        }
}

void Mpcc::updateState(const std::vector<double>& newState) {
    state = newState;
}

std::vector<double> Mpcc::getAction() {
    return action;
}

std::vector<std::vector<double> > Mpcc::getActions() {
    return actions;
}

void Mpcc::doMpc() {
    std::vector<double> controlsTMP = get_all_controls(state);
    // Update actions
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
            actions[j][i] =  controlsTMP[i*p + j];
        }
    }
    // Update action value at first sampling instant
    for (int i = 0; i < p; ++i) {
        action[i] = actions[i][0];
    }
}
