/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "bond_bpm_cbm.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

static constexpr double EPSILON = 1e-10;

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondBPMCBM::BondBPMCBM(LAMMPS *_lmp) :
    BondBPM(_lmp), k(nullptr), ecrit(nullptr), gamma(nullptr),
    id_fix_property_atom(nullptr)
{
  partial_flag = 1;
  smooth_flag = 1;
  normalize_flag = 0;
  writedata = 0;

  nhistory = 1;
  id_fix_bond_history = utils::strdup("HISTORY_BPM_CBM");

  single_extra = 1;
  svector = new double[1];

  comm_forward = atom->bond_per_atom + 1;
}

/* ---------------------------------------------------------------------- */

BondBPMCBM::~BondBPMCBM()
{
  delete[] svector;

  if (id_fix_property_atom) {
    modify->delete_fix(id_fix_property_atom);
    delete[] id_fix_property_atom;
  }

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(ecrit);
    memory->destroy(gamma);
  }
}

/* ----------------------------------------------------------------------
  Store data for all bonds, called once
------------------------------------------------------------------------- */

void BondBPMCBM::store_data()
{
  int i1, i2, i3, i4, i, j, m, n, n2, n3, n4, type, shared;
  double delx, dely, delz, r;
  double **x = atom->x;

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  // initialization of atom/nodal volumes
  double *vol = atom->dvector[index_vol];

  // send bond data to ghosts for identification of tets
  comm->forward_comm(this);

  // Identify tetrahedrons, order tag[i1] < tag[i2] < ...
  tagint tag1, tag2, tag3, tag4;
  int ntet = 0;
  for (i1 = 0; i1 < nlocal; i1++) {

    tag1 = tag[i1];

    // Find 2nd node
    for (n2 = 0; n2 < num_bond[i1]; n2++) {
      tag2 = bond_atom[i1][n2];
      if (tag2 < tag1) continue;

      i2 = atom->map(tag2);
      if (i2 == -1) error->one(FLERR, "Missing bond atom");

      // Find 3rd node

      for (n3 = 0; n3 < num_bond[i1]; n3++) {
        tag3 = bond_atom[i1][n3];
        if (tag3 < tag2) continue;

        // Check all mutually bonded
        shared = 0;
        for (n = 0; n < num_bond[i2]; n++) {
          if (bond_atom[i2][n] == tag3) {
            shared += 1;
            break;
          }
        }

        if (shared != 1) continue;

        i3 = atom->map(tag3);
        if (i3 == -1) error->one(FLERR, "Missing bond atom");

        // Find 4th node

        for (n4 = 0; n4 < num_bond[i1]; n4++) {
          tag4 = bond_atom[i1][n4];
          if (tag4 < tag3) continue;

          // Check all mutually bonded
          shared = 0;
          for (n = 0; n < num_bond[i2]; n++) {
            if (bond_atom[i2][n] == tag4) {
              shared += 1;
              break;
            }
          }
          for (n = 0; n < num_bond[i3]; n++) {
            if (bond_atom[i3][n] == tag4) {
              shared += 1;
              break;
            }
          }
          if (shared != 2) continue;

          i4 = atom->map(tag4);
          if (i4 == -1) error->one(FLERR, "Missing bond atom");

          printf("Tet %d between %d %d %d %d, n %d %d %d %d\n", ntet, tag1, tag2, tag3, tag4, n, n2, n3, n4);
          ntet += 1;
        }
      }
    }
  }


  for (i = 0; i < atom->nlocal; i++) {
    for (m = 0; m < atom->num_bond[i]; m++) {
      type = bond_type[i][m];

      // Skip if bond was turned off
      if (type <= 0) continue;

      // map to find index n
      j = atom->map(atom->bond_atom[i][m]);
      if (j == -1) error->one(FLERR, "Atom missing in BPM bond");

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];

      // Get closest image in case bonded with ghost
      domain->minimum_image(FLERR, delx, dely, delz);
      r = sqrt(delx * delx + dely * dely + delz * delz);

      fix_bond_history->update_atom_value(i, m, 0, r);
    }
  }
}

/* ---------------------------------------------------------------------- */

void BondBPMCBM::compute(int eflag, int vflag)
{
  pre_compute();

  int i1, i2, itmp, n, type, shared;
  tagint tag1, tag2, tag3, tag4;
  double delx, dely, delz, delvx, delvy, delvz;
  double e, rsq, r, r0, rinv, smooth, fbond, ebond, dot;

  ev_init(eflag, vflag);

  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double dim = domain->dimension;
  double invdim = 1.0 / dim;

  double **bondstore = fix_bond_history->bondstore;
  const bool allow_breaks = (update->setupflag == 0) && break_flag;

  // Calculate forces, currently just springs

  for (n = 0; n < nbondlist; n++) {

    // skip bond if already broken
    if (bondlist[n][2] <= 0) continue;

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    // Ensure pair is always ordered to ensure numerical operations
    // are identical to minimize the possibility that a bond straddling
    // an mpi grid (newton off) doesn't break on one proc but not the other
    if (tag[i2] < tag[i1]) {
      itmp = i1;
      i1 = i2;
      i2 = itmp;
    }

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    // If bond hasn't been set (should be initialized to zero)
    r0 = bondstore[n][0];
    if (r0 < EPSILON || std::isnan(r0)) {
      r0 = bondstore[n][0] = r;
      process_new(n, i1, i2);
    }

    e = (r - r0) / r0;

    if ((fabs(e) > ecrit[type]) && allow_breaks) {
      bondlist[n][2] = 0;
      process_broken(i1, i2);
      continue;
    }

    rinv = 1.0 / r;
    if (normalize_flag)
      fbond = -k[type] * e;
    else
      fbond = -k[type] * (r - r0);

    if (eflag) ebond = -0.5 * fbond * (r - r0);

    delvx = v[i1][0] - v[i2][0];
    delvy = v[i1][1] - v[i2][1];
    delvz = v[i1][2] - v[i2][2];
    dot = delx * delvx + dely * delvy + delz * delvz;
    fbond -= gamma[type] * dot * rinv;
    fbond *= rinv;

    if (smooth_flag) {
      smooth = (r - r0) / (r0 * ecrit[type]);
      smooth *= smooth;
      smooth *= smooth;
      smooth *= smooth;
      smooth = 1 - smooth;
      fbond *= smooth;
    }

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx * fbond;
      f[i1][1] += dely * fbond;
      f[i1][2] += delz * fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx * fbond;
      f[i2][1] -= dely * fbond;
      f[i2][2] -= delz * fbond;
    }

    if (evflag) ev_tally(i1, i2, nlocal, newton_bond, ebond, fbond, delx, dely, delz);
  }

  post_compute();
}

/* ---------------------------------------------------------------------- */

void BondBPMCBM::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(k, np1, "bond:k");
  memory->create(ecrit, np1, "bond:ecrit");
  memory->create(gamma, np1, "bond:gamma");

  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondBPMCBM::coeff(int narg, char **arg)
{
  if ((!volume_flag && narg != 4) || (volume_flag && narg != 5))
    error->all(FLERR, "Incorrect args for bond coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double k_one = utils::numeric(FLERR, arg[1], false, lmp);
  double ecrit_one = utils::numeric(FLERR, arg[2], false, lmp);
  double gamma_one = utils::numeric(FLERR, arg[3], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    ecrit[i] = ecrit_one;
    gamma[i] = gamma_one;
    setflag[i] = 1;
    count++;

    if (1.0 + ecrit[i] > max_stretch) max_stretch = 1.0 + ecrit[i];
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   check for correct settings and create fix
------------------------------------------------------------------------- */

void BondBPMCBM::init_style()
{
  BondBPM::init_style();

  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Bond cbm requires ghost atoms store velocity");

  if (force->newton_bond == 1)
    error->all(FLERR, "Bond CBM requires newton off");

  if (domain->dimension == 2)
    error->all(FLERR, "Bond CBM currently only works in 3D");


  if (!id_fix_property_atom) {
    id_fix_property_atom = utils::strdup("BOND_BPM_CBM_FIX_PROPERTY_ATOM");
    modify->add_fix(fmt::format("{} all property/atom d_vol ghost yes writedata no",
                                id_fix_property_atom));

    int tmp1 = 0, tmp2 = 0;
    index_vol = atom->find_custom("vol", tmp1, tmp2);
  }
}

/* ---------------------------------------------------------------------- */

void BondBPMCBM::settings(int narg, char **arg)
{
  BondBPM::settings(narg, arg);

  int iarg;
  for (std::size_t i = 0; i < leftover_iarg.size(); i++) {
    iarg = leftover_iarg[i];
    if (strcmp(arg[iarg], "smooth") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal bond bpm command, missing option for smooth");
      smooth_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      i += 1;
    } else if (strcmp(arg[iarg], "normalize") == 0) {
      if (iarg + 1 > narg)
        error->all(FLERR, "Illegal bond bpm command, missing option for normalize");
      normalize_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      i += 1;
    } else {
      error->all(FLERR, "Illegal bond bpm command, invalid argument {}", arg[iarg]);
    }
  }

  if (smooth_flag && !break_flag)
    error->all(FLERR, "Illegal bond bpm command, must turn off smoothing with break no option");
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondBPMCBM::write_restart(FILE *fp)
{
  BondBPM::write_restart(fp);
  write_restart_settings(fp);

  fwrite(&k[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&ecrit[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&gamma[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondBPMCBM::read_restart(FILE *fp)
{
  BondBPM::read_restart(fp);
  read_restart_settings(fp);
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &k[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &ecrit[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &gamma[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&k[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&ecrit[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&gamma[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void BondBPMCBM::write_restart_settings(FILE *fp)
{
  fwrite(&smooth_flag, sizeof(int), 1, fp);
  fwrite(&normalize_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void BondBPMCBM::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &smooth_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &normalize_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&smooth_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&normalize_flag, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

double BondBPMCBM::single(int type, double rsq, int i, int j, double &fforce)
{
  if (type <= 0) return 0.0;

  double r0;
  for (int n = 0; n < atom->num_bond[i]; n++) {
    if (atom->bond_atom[i][n] == atom->tag[j]) r0 = fix_bond_history->get_atom_value(i, n, 0);
  }

  double r = sqrt(rsq);
  double rinv = 1.0 / r;
  double e = (r - r0) / r0;

  if (normalize_flag)
    fforce = -k[type] * e;
  else
    fforce = -k[type] * (r - r0);

  double ebond = -0.5 * fforce * (r - r0);

  double **x = atom->x;
  double **v = atom->v;
  double delx = x[i][0] - x[j][0];
  double dely = x[i][1] - x[j][1];
  double delz = x[i][2] - x[j][2];
  double delvx = v[i][0] - v[j][0];
  double delvy = v[i][1] - v[j][1];
  double delvz = v[i][2] - v[j][2];
  double dot = delx * delvx + dely * delvy + delz * delvz;
  fforce -= gamma[type] * dot * rinv;
  fforce *= rinv;

  if (smooth_flag) {
    double smooth = (r - r0) / (r0 * ecrit[type]);
    smooth *= smooth;
    smooth *= smooth;
    smooth *= smooth;
    smooth = 1 - smooth;
    fforce *= smooth;
  }

  // set single_extra quantities

  svector[0] = r0;

  return ebond;
}

/* ---------------------------------------------------------------------- */

int BondBPMCBM::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    buf[m++] = ubuf(num_bond[i]).d;
    for (int nb = 0; nb < num_bond[i]; nb++)
      buf[m++] = ubuf(bond_atom[i][nb]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void BondBPMCBM::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  int *num_bond = atom->num_bond;
  tagint **bond_atom = atom->bond_atom;
  int last = first + n;
  for (int i = first; i < last; i++) {
    num_bond[i] = ubuf(buf[m++]).i;
    for (int nb = 0; nb < num_bond[i]; nb++)
      bond_atom[i][nb] = (tagint) ubuf(buf[m++]).i;
  }
}
