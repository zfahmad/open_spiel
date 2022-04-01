# To enable C++ Torch AlphaZero, you will need to set OPEN_SPIEL_BUILD_WITH_LIBTORCH.
if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  if(NOT OPEN_SPIEL_BUILD_WITH_LIBNOP)
    message(FATAL_ERROR
      "alpha_zero_torch requires libnop (OPEN_SPIEL_BUILD_WITH_LIBNOP)")
  endif()

  add_library (alpha_zero_torch OBJECT
    alpha_zero.h
    alpha_zero.cc
    device_manager.h
    model.h
    model.cc
    vpevaluator.h
    vpevaluator.cc
    vpnet.h
    vpnet.cc
    uct.h
    uct.cc
    puct.h
    puct.cc
    lts.h
    lts.cc
    bf_lts.h
    bf_lts.cc
    bf2lts.h
    bf2lts.cc
    lts_self_play.h
    lts_self_play.cc)

  target_include_directories (alpha_zero_torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(torch_model_test model_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_model_test torch_model_test)

  add_executable(torch_vpnet_test vpnet_test.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)
  add_test(torch_vpnet_test torch_vpnet_test)

  add_executable(state_evaluator state_evaluator.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(negamax negamax.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(alpha_beta_negamax alpha_beta_negamax.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(id_ab_negamax id_ab_negamax.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(two_lts two_lts.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(evaluate evaluate.cc ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

            # add_executable(play play.cc ${OPEN_SPIEL_OBJECTS}
            #               $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(test_bf2lts test_bf2lts.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  add_executable(lts_run lts_run.cc ${OPEN_SPIEL_OBJECTS}
          $<TARGET_OBJECTS:alpha_zero_torch> $<TARGET_OBJECTS:tests>)

  target_link_libraries (alpha_zero_torch ${TORCH_LIBRARIES})
  target_link_libraries (torch_model_test ${TORCH_LIBRARIES})
  target_link_libraries (torch_vpnet_test ${TORCH_LIBRARIES})
  target_link_libraries (state_evaluator ${TORCH_LIBRARIES})
  target_link_libraries (negamax ${TORCH_LIBRARIES})
  target_link_libraries (alpha_beta_negamax ${TORCH_LIBRARIES})
  target_link_libraries (id_ab_negamax ${TORCH_LIBRARIES})
  target_link_libraries (two_lts ${TORCH_LIBRARIES})
  target_link_libraries (evaluate ${TORCH_LIBRARIES})
  # target_link_libraries (play ${TORCH_LIBRARIES})
  target_link_libraries (test_bf2lts ${TORCH_LIBRARIES})
  target_link_libraries (lts_run ${TORCH_LIBRARIES})
endif ()
