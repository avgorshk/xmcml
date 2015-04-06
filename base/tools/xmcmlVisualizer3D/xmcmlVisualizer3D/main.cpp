#include <stdlib.h>
#include "xmcml_demo.h"

int main(int argc, char* argv[])
{
    xmcml_demo demo(argc, argv, "eXtended Monte Carlo Modeling of Light transport");
    demo.run();
}