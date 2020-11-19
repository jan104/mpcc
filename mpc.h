#pragma once
#include <casadi/casadi.hpp>

using namespace casadi;

class Mpcc{
public:
    size_t n;
    size_t p;
    size_t N;
    SX x;
    SX u;
    SX A;
    SX B;
    SX Q;
    SX R;
    SX xDot;
    Function system;
    Function stage_cost_fcn;
    Function terminal_cost_fcn;
    std::vector<double> state;
    std::vector<double> action;
    std::vector<std::vector<double> > actions;
    std::vector<double> lb_x;
    std::vector<double> ub_x;
    std::vector<double> lb_u;
    std::vector<double> ub_u;
    SX X;
    SX U;
    Function mpc_solver;
    std::map<std::string, DM> arg, mpc_res;
    // Used to store the FULL mpc result (containts x and u)
    std::vector<double> tmp;
    // This is used to locate the first u in the resulting vector
    size_t firstU;
    // contains the target state
    std::vector<double> target;
    // single optimization variable
    SX ref;
    // for formulating the problem
    SX ref_k;
    // ref for every N
    SX REF;
    // target state for every N
    std::vector<double> targetTrajectory;
    Mpcc();

    Mpcc(std::vector<std::vector<double>> AMatrix, 
        std::vector<std::vector<double>> BMatrix,
        std::vector<std::vector<double>> QMatrix,
        std::vector<std::vector<double>> RMatrix,
        std::vector<double> lowerbound_x,
        std::vector<double> upperbound_x,
        std::vector<double> lowerbound_u,
        std::vector<double> upperbound_u,
        std::vector<double> target,
        size_t horizon);

    void updateTarget(const std::vector<double>& newTarget);
    void updateState(const std::vector<double>& newState);
    std::vector<double> getAction();
    std::vector<std::vector<double> > getActions();
    void doMpc();

private:
    SX stage_cost;
    SX terminal_cost;
    SX x_k;
    SX x_k_next;
    SX x_k_next_calc;
    SX u_k;
    SX J;
    std::vector<double> zeroVec;
    std::vector<SX> g;
    std::vector<double> lb_g;
    std::vector<double> ub_g;
    std::vector<double> lb_X;
    std::vector<double> ub_X;
    std::vector<double> lb_U;
    std::vector<double> ub_U;
    SX x_terminal;
    // These are used for the nlp solver. New names as the dimensions 
    // or types have to be adjusted
    SX lbx;
    SX ubx;
    SX xx;
    SX lbg;
    SX ubg;
    SX gg;
    // Used to pass options to nlpsol
    Dict opts_dict;
    // This will contain the information for the nlpsolver
    SXDict prob;
    std::vector<double> get_all_controls(const std::vector<double>& currentState);
};
