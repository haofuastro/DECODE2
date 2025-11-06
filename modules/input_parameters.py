"""
@ input_parameters.py

Written by Hao Fu

The goal of this module is to read the input parameters from file.
"""

import numpy as np
import getopt
import sys
import os
import ast




def set_statement(statement):

    if statement == "yes":
        return 1
    elif statement == "no":
        return 0
    else:
        ErrMsg = "  /!\    Error in the input parameter file:\n / ! \   statement error.\n"
        sys.exit(ErrMsg)


def bool_to_binary(statement):

    if statement:
        return 1
    elif not statement:
        return 0
    else:
        ErrMsg = "  /!\    Error in the input parameter file:\n / ! \   statement error.\n"
        sys.exit(ErrMsg)



class input_parameters_class:

    """
    Return parameters:
    z0, reduced_catalogue, reduce_fraction, sigma_sfr, delay, luminosity_function, halo_quenching, logMhDekkel, scatter_mhlog_crit, mergers_quenching, include_mergers, merger_tree, mratio_threshold, include_sats_starformation, include_sats_quenching, include_sats_stripping, include_SNfeedback, logMh_SNfeedback, f_discregrowth, add_disc_inst
    """

    def __init__(self, z0, reduced_catalogue, reduce_fraction, logmbh_seed, sigma_sfr, sfr_delay_alpha, delay, luminosity_function, halo_quenching, logMhDekkel, scatter_mhlog_crit, mergers_quenching, include_mergers, merger_tree, mratio_threshold, include_sats_starformation, include_sats_quenching, include_sats_stripping, include_SNfeedback, logMh_SNfeedback, f_discregrowth, add_disc_inst):

        self.z0 = z0
        self.reduced_catalogue = reduced_catalogue
        self.reduce_fraction = reduce_fraction
        self.logmbh_seed = logmbh_seed
        self.sigma_sfr = sigma_sfr
        self.sfr_delay_alpha = sfr_delay_alpha
        self.delay = delay
        self.luminosity_function = halo_quenching
        self.halo_quenching = halo_quenching
        self.logMhDekkel = logMhDekkel
        self.scatter_mhlog_crit = scatter_mhlog_crit
        self.mergers_quenching = mergers_quenching
        self.include_mergers = include_mergers
        self.merger_tree = merger_tree
        self.mratio_threshold = mratio_threshold
        self.include_sats_starformation = include_sats_starformation
        self.include_sats_quenching = include_sats_quenching
        self.include_sats_stripping = include_sats_stripping
        self.include_SNfeedback = include_SNfeedback
        self.logMh_SNfeedback = logMh_SNfeedback
        self.f_discregrowth = f_discregrowth
        self.add_disc_inst = add_disc_inst



def read_input_parameters(input_filename):

    print("Reading parameters from file:\n{}\n".format(input_filename))
    with open(input_filename) as input_file:
        input_lines = input_file.readlines()

    for i in range(len(input_lines)):

        line = input_lines[i].split()

        if len(line)!=0 and line[0][0]!="#":

            if "z0" in line: z0 = float(line[2])
            elif "reduced_catalogue" in line: reduced_catalogue = line[2]
            elif "reduce_fraction" in line: reduce_fraction = float(line[2])
            elif "logmbh_seed" in line: logmbh_seed = float(line[2])
            elif "sigma_sfr" in line: sigma_sfr = float(line[2])
            elif "sfr_delay_alpha" in line: sfr_delay_alpha = float(line[2])
            elif "delay" in line: delay = line[2]
            elif "luminosity_function" in line: luminosity_function = line[2]
            elif "halo_quenching" in line: halo_quenching = line[2]
            elif "logMhDekkel" in line: logMhDekkel = float(line[2])
            elif "scatter_mhlog_crit" in line: scatter_mhlog_crit = float(line[2])
            elif "mergers_quenching" in line: mergers_quenching = line[2]
            elif "include_mergers" in line: include_mergers = line[2]
            elif "merger_tree" in line: merger_tree = line[2]
            elif "mratio_threshold" in line: mratio_threshold = float(line[2])
            elif "include_sats_starformation" in line: include_sats_starformation = line[2]
            elif "include_sats_quenching" in line: include_sats_quenching = line[2]
            elif "include_sats_stripping" in line: include_sats_stripping = line[2]
            elif "include_SNfeedback" in line: include_SNfeedback = line[2]
            elif "logMh_SNfeedback" in line: logMh_SNfeedback = float(line[2])
            elif "f_discregrowth" in line: f_discregrowth = float(line[2])
            elif "add_disc_inst" in line: add_disc_inst = line[2]

    try:
        z0
    except:
        ErrMessage = "  /!\    Error in the input parameter file:\n / ! \   redshift missing\n"
        sys.exit(ErrMessage)

    try:
        if reduced_catalogue == "yes":
            reduced_catalogue = True
        elif reduced_catalogue == "no":
            reduced_catalogue = False
    except:
        reduced_catalogue = False

    if reduced_catalogue:
        try:
            reduce_fraction
        except:
            ErrMessage = "  /!\    Error in the input parameter file:\n / ! \   reduce fraction missing\n"
            sys.exit(ErrMessage)

    try:
        sigma_sfr
    except:
        sigma_sfr = 0.4
        print("  /!\    Scatter in SFR at given HAR not given.\n / ! \   Set to default = 0.4.\n")

    try:
        if delay == "yes":
            delay = True
        elif delay == "no":
            delay = False
    except:
        delay = False

    try:
        luminosity_function
    except:
        luminosity_function = "Mancuso+2016"

    try:
        if halo_quenching == "yes":
            halo_quenching = True
        elif halo_quenching == "no":
            halo_quenching = False
    except:
        halo_quenching = True

    try:
        logMhDekkel
    except:
        logMhDekkel = 12.24
        print("  /!\    Critical halo mass not given.\n / ! \   Set to default = 12.24.\n")

    try:
        scatter_mhlog_crit
    except:
        scatter_mhlog_crit = 0.4
        print("  /!\    Scatter in critical halo mass not given.\n / ! \   Set to default = 0.4.\n")

    try:
        if mergers_quenching == "yes":
            mergers_quenching = True
        elif mergers_quenching == "no":
            mergers_quenching = False
    except:
        mergers_quenching = False

    try:
        if include_mergers == "yes":
            include_mergers = True
        elif include_mergers == "no":
            include_mergers = False
    except:
        include_mergers = True

    try:
        merger_tree
    except:
        merger_tree = "SatGen"

    try:
        mratio_threshold
    except:
        mratio_threshold = 0.25

    try:
        if include_sats_starformation == "yes":
            include_sats_starformation = True
        elif include_sats_starformation == "no":
            include_sats_starformation = False
    except:
        include_sats_starformation = False

    try:
        if include_sats_quenching == "yes":
            include_sats_quenching = True
        elif include_sats_quenching == "no":
            include_sats_quenching = False
    except:
        include_sats_quenching = False

    try:
        if include_sats_stripping == "yes":
            include_sats_stripping = True
        elif include_sats_stripping == "no":
            include_sats_stripping = False
    except:
        include_sats_stripping = False

    try:
        if include_SNfeedback == "yes":
            include_SNfeedback = True
        elif include_SNfeedback == "no":
            include_SNfeedback = False
    except:
        include_SNfeedback = False

    try:
        logMh_SNfeedback
    except:
        logMh_SNfeedback = 11.

    try:
        f_discregrowth
    except:
        f_discregrowth = 0.5

    try:
        if add_disc_inst == "yes":
            add_disc_inst = True
        elif add_disc_inst == "no":
            add_disc_inst = False
    except:
        add_disc_inst = False

    input_params = input_parameters_class(z0, reduced_catalogue, reduce_fraction, logmbh_seed, sigma_sfr, sfr_delay_alpha, delay, luminosity_function, halo_quenching, logMhDekkel, scatter_mhlog_crit, mergers_quenching, include_mergers, merger_tree, mratio_threshold, include_sats_starformation, include_sats_quenching, include_sats_stripping, include_SNfeedback, logMh_SNfeedback, f_discregrowth, add_disc_inst)

    return input_params
