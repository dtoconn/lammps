/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(bpm/cbm,BondBPMCBM);
// clang-format on
#else

#ifndef LMP_BOND_BPM_CBM_H
#define LMP_BOND_BPM_CBM_H

#include "bond_bpm.h"

namespace LAMMPS_NS {

class BondBPMCBM : public BondBPM {
 public:
  BondBPMCBM(class LAMMPS *);
  ~BondBPMCBM() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  void settings(int, char **) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double single(int, double, int, int, double &) override;

 protected:
  double *k, *ecrit, *gamma;
  int smooth_flag, normalize_flag, volume_flag, index_vol;
  char *id_fix_property_atom;

  void allocate();
  void store_data() override;
};

}    // namespace LAMMPS_NS

#endif
#endif
