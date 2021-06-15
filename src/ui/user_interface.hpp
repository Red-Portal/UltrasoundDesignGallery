
/*
 * Copyright (C) 2021  Ray Kim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __US_GALLERY_UI_HPP__
#define __US_GALLERY_UI_HPP__

#include <optional>

#include "video_player.hpp"
#include "line_search.hpp"
#include "optimization_manager.hpp"

namespace usdg
{
  class UserInterface
  {
/*
 *  Internal state machine:
 *
 *                        ┌─reset──┐
 *                        │        │
 *                        │        ▼
 *  ┌────┐               ┌┴────────┐            ┌───────────┐             ┌──────────┐
 *  │idle├──────────────►│rendering├─selected──►│optimizing │─optimized──►│optimized │
 *  └────┘               └────┬────┘            └───────────┘             └─────┬────┘
 *     ▲                      │ ▲                                               │
 *     └─close player window──┘ └───────────────────────────────────────────────┘
 * 
 *  Idle:       line search module does not work, no video player window
 *  Rendering:  line search module works, select button is enabled
 *  optimizing: select button is disabled, reset does not work (error message?)
 *  optimized:  select button is enabled back, reset does not work
 *
 *  Note: "Reset" is not part of the actual state machine execution flow.
 */

    enum class UIState
    {
      idle,
      rendering,
      optimizing,
      optimized
    };
    
  private:
    std::optional<usdg::VideoPlayer> _video_player;
    usdg::LineSearch                 _linesearch;
    UIState                          _state;
    usdg::OptimizationManager        _opt_manager;

    void render_menubar();

  public:
    void state_render();

    void state_action();

    void state_transition();

    friend std::ostream& operator<<(std::ostream& os, UserInterface const&);
  };
}

#endif
