KERNEL_BIN="./kernel_bose_linear_2.out"

echo "--- Running Moscal for Kernel Functions at Temperature: ${tt} ---"
echo "--- * getting kernal projection to state rho_{00}"
cp 00.input.json input.json
${KERNEL_BIN} >00.kernel.log
# cleaning up
mv prop-kernel-eq.dat 00.prop-kernel-eq.dat
rm curr.dat
echo "--- * getting kernal projection to state rho_{11}"
cp 01.input.json input.json
${KERNEL_BIN} >01.kernel.log
# cleaning up
mv prop-kernel-eq.dat 01.prop-kernel-eq.dat
rm curr.dat input.json
